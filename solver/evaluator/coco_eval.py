from detectron2.evaluation import inference_on_dataset, COCOEvaluator

class CocoDefaultEvaluator:
    def __init__(self, dataset_name="test", output_dir="outputs", allow_cached_coco=False):
        self.forward = inference_on_dataset
        self.evaluator = COCOEvaluator(
            dataset_name=dataset_name,
            allow_cached_coco=allow_cached_coco,
            output_dir=output_dir
        )

    def __call__(self, model, test_loader):
        return self.forward(model, test_loader, self.evaluator)