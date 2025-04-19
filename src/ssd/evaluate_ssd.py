def evaluate_ssd(model_path, data_yaml, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    with mlflow.start_run(run_name="SSD_Evaluation"):
        mlflow.log_params({
            "model_path": model_path, "data_yaml": data_yaml, "num_classes": num_classes,
            "device": device, "model_name": "ssdlite320_mobilenet_v3_large"
        })

        # Model setup
        model = ssdlite320_mobilenet_v3_large(weights=None)
        in_channels = [module[0][0].in_channels for module in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes,
            norm_layer=partial(torch.nn.BatchNorm2d, eps=1e-3, momentum=0.03)
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        # Use test split for final evaluation
        test_dataset = SSDDataset(data_yaml, split='test', transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

        metric = MeanAveragePrecision(max_detection_thresholds=[1, 100, 500])
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, targets in test_loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(images)
                preds = [{'boxes': o['boxes'].cpu(), 'scores': o['scores'].cpu(), 'labels': o['labels'].cpu()} for o in outputs]
                targets_cpu = [{'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()} for t in targets]
                metric.update(preds, targets_cpu)
                all_preds.extend(preds)
                all_targets.extend(targets_cpu)
            map_dict = metric.compute()
            mAP = map_dict['map'].item()
            mAP50 = map_dict['map_50'].item()
            # Compute precision and recall manually
            tp, fp, fn = 0, 0, 0
            conf_threshold = 0.5
            iou_threshold = 0.5
            for pred, target in zip(all_preds, all_targets):
                pred_boxes = pred['boxes'][pred['scores'] >= conf_threshold]
                pred_labels = pred['labels'][pred['scores'] >= conf_threshold]
                target_boxes = target['boxes']
                target_labels = target['labels']
                if len(pred_boxes) == 0 and len(target_boxes) > 0:
                    fn += len(target_boxes)
                    continue
                if len(target_boxes) == 0 and len(pred_boxes) > 0:
                    fp += len(pred_boxes)
                    continue
                if len(pred_boxes) == 0 and len(target_boxes) == 0:
                    continue
                ious = box_iou(pred_boxes, target_boxes)
                for i, pred_label in enumerate(pred_labels):
                    matched = False
                    for j, target_label in enumerate(target_labels):
                        if pred_label == target_label and ious[i, j] >= iou_threshold:
                            tp += 1
                            matched = True
                            break
                    if not matched:
                        fp += 1
                fn += len(target_boxes) - (ious.max(dim=0)[0] >= iou_threshold).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        mlflow.log_metrics({"mAP": mAP, "mAP50": mAP50, "precision": precision, "recall": recall})
        mlflow.pytorch.log_model(model, "evaluated_model")

        print(f"mAP: {mAP}, mAP50: {mAP50}, Precision: {precision}, Recall: {recall}")


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union