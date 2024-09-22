# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
import cv2
import os


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results


if __name__ == '__main__':
    def some_function_that_uses_yolo():
        from ultralytics import YOLO
        # 初始化模型
        model_path = 'E:\\myfile\\ship-classification-yolo\\ultralytics-main-yolov8\\ultralytics\\models\yolo\\detect\\yolov8x.pt'  # 替换为你自己的模型路径
        model = YOLO(model_path)

        # 设置预测参数
        args = dict(conf=0.25, iou=0.5, agnostic_nms=False, max_det=100, classes=None)

        # 初始化预测器
        predictor = DetectionPredictor(overrides=args)
        predictor.setup_model(model)

        # 指定输入图像路径
        image_path = 'path/to/your/image.jpg'  # 替换为你的图像路径

        # 进行预测
        results = predictor(image_path)

        # 保存结果
        output_dir = 'path/to/save/results'  # 替换为你想要保存结果的目录
        os.makedirs(output_dir, exist_ok=True)

        for i, result in enumerate(results):
            output_path = os.path.join(output_dir, f'result_{i}.jpg')
            result.save_txt(os.path.join(output_dir, f'result_{i}.txt'))
            result.save_crop(save_dir=output_dir)
            cv2.imwrite(output_path, result.plot())

        print(f'Results saved to {output_dir}')