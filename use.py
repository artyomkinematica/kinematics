import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from scipy import ndimage
from torch.nn import init


class MLP(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class CovarianceColliderLayer(nn.Module):
    def __init__(self, dim, features_dim, heads, covariance_depth, second_projection_dim, dropout):
        super().__init__()
        self.dim = dim
        self.features_dim = features_dim
        self.heads = heads
        self.head_dim = dim // heads
        self.covariance_depth = covariance_depth

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_features = nn.Linear(dim, features_dim * heads)
        self.feature_norm = nn.LayerNorm(features_dim)

        self.covariance_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(features_dim, second_projection_dim),
                nn.GELU(),
                nn.Linear(second_projection_dim, features_dim)
            ) for _ in range(covariance_depth)
        ])

        self.hierarchical_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(features_dim, features_dim),
                nn.LayerNorm(features_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(features_dim, features_dim)
            ) for _ in range(covariance_depth)
        ])

        self.hierarchical_weights = nn.Parameter(torch.ones(covariance_depth) / covariance_depth)
        self.attention_pool = nn.Linear(features_dim, 1)

        self.reproject_to_dim = nn.Linear(features_dim * heads, dim)
        self.output_norm = nn.LayerNorm(dim)
        self.output_mlp = MLP(dim, dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0.0)

    def compute_covariance_features(self, features):
        b, h, s, f = features.shape
        features_mean = features.mean(dim=2, keepdim=True)
        features_centered = features - features_mean
        covariance = torch.matmul(
            features_centered.transpose(2, 3),
            features_centered
        ) / max(s - 1, 1)
        return covariance

    def forward(self, x):
        b, s, d = x.shape
        h = self.heads

        qkv = self.to_qkv(x).reshape(b, s, 3, h, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)
        attended_v = torch.matmul(attention, v)

        features = self.to_features(x).reshape(b, s, h, self.features_dim).permute(0, 2, 1, 3)
        features = self.feature_norm(features)

        hierarchical_features = []
        current_features = features

        for depth in range(self.covariance_depth):
            covariance = self.compute_covariance_features(current_features)

            processed_covariance = self.covariance_projections[depth](
                covariance.reshape(b * h, self.features_dim, self.features_dim)
            ).reshape(b, h, self.features_dim, self.features_dim)

            covariance_features = processed_covariance.mean(dim=3)

            transformed = self.hierarchical_transforms[depth](
                covariance_features.reshape(b * h, self.features_dim)
            ).reshape(b, h, self.features_dim)

            hierarchical_features.append(transformed.unsqueeze(2))

            if depth < self.covariance_depth - 1:
                attention_weights = self.attention_pool(transformed).squeeze(-1)
                attention_weights = F.softmax(attention_weights, dim=-1)
                weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
                current_features = weights * current_features

        hierarchical_weights = F.softmax(self.hierarchical_weights, dim=0)

        combined_features = torch.zeros_like(features)
        for idx, (weight, feat) in enumerate(zip(hierarchical_weights, hierarchical_features)):
            combined_features = combined_features + weight * feat

        combined_features = combined_features.permute(0, 2, 1, 3).reshape(b, s, h * self.features_dim)

        reprojected = self.reproject_to_dim(combined_features)

        attended_output = attended_v.permute(0, 2, 1, 3).reshape(b, s, d)

        x = x + attended_output + reprojected
        x = self.output_norm(x)
        x = x + self.output_mlp(x)

        return x


class ImageClassificationModel(nn.Module):
    def __init__(self, image_size=40, patch_size=4, num_classes=6, dim=128, depth=3, heads=4, features_dim=64,
                 covariance_depth=2, second_projection_dim=32, dropout=0.2):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

        self.layers = nn.ModuleList([
            CovarianceColliderLayer(dim, features_dim, heads, covariance_depth, second_projection_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        init.normal_(self.cls_token, std=0.02)
        init.normal_(self.pos_embed, std=0.02)
        init.xavier_uniform_(self.head.weight)
        init.constant_(self.head.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.head(x[:, 0])


class ObjectDetector:
    def __init__(self, classifier_model_path='img_cognition_best_stable.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.classifier_model = ImageClassificationModel(
            image_size=40,
            patch_size=4,
            num_classes=6,
            dim=256,
            depth=5,
            heads=8,
            features_dim=128,
            covariance_depth=10,
            second_projection_dim=64,
            dropout=0.2
        ).to(self.device)
        print(sum(p.numel() for p in self.classifier_model.parameters() if p.requires_grad))
        self.classifier_model.load_state_dict(torch.load(classifier_model_path, map_location=self.device, weights_only=True))
        self.classifier_model.eval()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classifier_transform = T.Compose([
            T.Resize((40, 40)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def detect_objects(self, image_path):
        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size
        gray = image.convert('L')
        arr = np.array(gray)

        smoothed = ndimage.gaussian_filter(arr, sigma=2)
        binary = smoothed < 90

        struct = ndimage.generate_binary_structure(2, 2)
        closed = ndimage.binary_closing(binary, structure=struct, iterations=2)
        labeled_im, num = ndimage.label(closed.astype(int))

        centers = []
        if num > 0:
            component_sizes = ndimage.sum(np.ones_like(labeled_im), labeled_im, index=np.arange(1, num + 1))
            min_area = 100
            valid_labels = [i for i, size in enumerate(component_sizes, start=1) if size >= min_area]
            if valid_labels:
                centers = ndimage.center_of_mass(np.ones_like(labeled_im), labeled_im, index=valid_labels)
                centers = [(int(round(cy)), int(round(cx))) for (cy, cx) in centers if not (np.isnan(cy) or np.isnan(cx))]

        detections = []
        for cy, cx in centers:
            crop_size = 140
            half = crop_size // 2
            left, top = cx - half, cy - half
            right, bottom = cx + half, cy + half

            pad_left = max(0, -left)
            pad_top = max(0, -top)
            pad_right = max(0, right - orig_w)
            pad_bottom = max(0, bottom - orig_h)

            if any(p > 0 for p in [pad_left, pad_top, pad_right, pad_bottom]):
                new_w = orig_w + pad_left + pad_right
                new_h = orig_h + pad_top + pad_bottom
                padded = Image.new('RGB', (new_w, new_h), (0, 0, 0))
                padded.paste(image, (pad_left, pad_top))
                nx, ny = cx + pad_left, cy + pad_top
                crop = padded.crop((nx - half, ny - half, nx + half, ny + half))
            else:
                crop = image.crop((left, top, right, bottom))

            if crop.size[0] < 40 or crop.size[1] < 40:
                continue
            crop = crop.resize((40, 40), Image.BILINEAR)
            detections.append({'center': (cx, cy), 'crop': crop})

        return detections, image

    def classify_objects(self, detections):
        results = []
        for det in detections:
            tensor = self.classifier_transform(det['crop']).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.classifier_model(tensor)
                probs = F.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
            results.append({
                'center': det['center'],
                'class': pred.item(),
                'confidence': conf.item()
            })
        return results

    def process_image(self, image_path):
        detections, original_image = self.detect_objects(image_path)
        results = self.classify_objects(detections)
        return {
            'image_path': image_path,
            'original_size': original_image.size,
            'detections': results
        }


def main():
    detector = ObjectDetector(classifier_model_path='img_cognition_best_stable.pth')
    image_path = "WIN_20260110_12_53_10_Pro.jpg"
    if not os.path.exists(image_path):
        print("Файл не найден!")
        return

    print(f"Обработка изображения: {image_path}")
    result = detector.process_image(image_path)

    print(f"\nРезультаты для {result['image_path']}:")
    print(f"Размер изображения: {result['original_size']}")
    print(f"Найдено объектов: {len(result['detections'])}")
    for i, det in enumerate(result['detections']):
        print(f"\nОбъект {i+1}:")
        print(f"  Координаты центра: {det['center']}")
        print(f"  Класс: {det['class']}")
        print(f"  Уверенность: {det['confidence']:.4f}")


if __name__ == '__main__':
    main()