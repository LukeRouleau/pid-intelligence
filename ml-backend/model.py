from typing import List, Dict, Optional
import uuid
import random
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Generate random bounding box predictions for object detection tasks
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        predictions = []
        
        # Classes that our mock detector can identify
        classes = ["person", "car", "bicycle", "motorcycle", "truck", "bus", "traffic_light", "stop_sign"]
        
        for task in tasks:
            # Get image data from task
            image_data = task.get('data', {})
            image_url = image_data.get('image', '')
            
            # Use get_local_path to access the image file
            if image_url:
                try:
                    local_path = get_local_path(image_url, task_id=task['id'])
                    print(f"Successfully accessed image at {local_path}")
                    
                    # In a real implementation, you would process the image here
                    # For our mock implementation, we'll still generate random boxes
                except Exception as e:
                    print(f"Error accessing image: {e}")
            
            # Generate 2-4 random bounding boxes
            num_boxes = random.randint(2, 4)
            results = []
            
            for _ in range(num_boxes):
                # Generate random box parameters
                x = random.randint(10, 80)  # x coordinate as percentage
                y = random.randint(10, 80)  # y coordinate as percentage
                width = random.randint(5, 20)  # width as percentage
                height = random.randint(5, 20)  # height as percentage
                confidence = round(random.uniform(0.7, 0.98), 2)  # confidence score
                class_name = random.choice(classes)  # random class
                
                # Create annotation in Label Studio format
                results.append({
                    "id": str(uuid.uuid4()),
                    "from_name": "label",  # This should match your labeling config
                    "to_name": "image",    # This should match your labeling config
                    "type": "rectanglelabels",
                    "score": confidence,
                    "value": {
                        "rectanglelabels": [class_name],
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height
                    }
                })
            
            predictions.append({
                "model_version": self.get("model_version"),
                "result": results
            })
        
        return ModelResponse(predictions=predictions)
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')
