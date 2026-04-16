# base_model.py

# Base class for VLM models
class BaseVLM:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def processor_function(self, batch):
        raise NotImplementedError

    def __call__(self, inputs, **gen_kwargs):
        outputs = self.model.generate(**inputs, **gen_kwargs, do_sample=False)
        return outputs

    def decode(self, outputs):
        return self.processor.decode(outputs[0], skip_special_tokens=True)
