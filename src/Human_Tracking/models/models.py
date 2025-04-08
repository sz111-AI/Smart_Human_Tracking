class BoxResult:
    def __init__(self, boxes_tensor, confs_tensor):
        self.boxes = boxes_tensor
        self.confs = confs_tensor

class DetectionResult:
    def __init__(self, boxes, conf):
        self.boxes = boxes
        self.conf = conf

    def __iter__(self):
        # This will make the DetectionResult iterable, returning a tuple of box and confidence
        for box, confidence in zip(self.boxes, self.conf):
            yield box, confidence
