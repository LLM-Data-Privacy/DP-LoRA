from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

class ServerCLIPModel:
    def __init__(self):
        """
        初始化模型和处理器。
        """
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def save_pretrained(self, save_directory):
        """
        将模型和处理器的配置和权重保存到指定目录。
        """
        # 保存模型
        self.model.save_pretrained(save_directory)
        # 保存处理器
        self.processor.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory):
        """
        从指定目录加载预训练的模型和处理器。
        """
        instance = cls.__new__(cls)
        instance.model = CLIPModel.from_pretrained(load_directory)
        instance.processor = CLIPProcessor.from_pretrained(load_directory)
        return instance