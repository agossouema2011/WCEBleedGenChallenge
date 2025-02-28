# WCEBleedGenChallenge
# The colorLab Team won the fourth consolation prize of the "AutoWCEBleedGenChallenge version 1".
<b>Notes:</b>
- Our github code is composed of two folders "Classification" and "Detection" containing respectively the classification and detection codes.
- All the codes are written in python.
- For the classification we used a pretrained <b>ResNet50</b> and finetuned it on our dataset. We chosed <b>ResNet50</b> because through the state-of-the art, it is more robust to error and achieves better performance on image classification for simpler dataset. 
- For the classification we used 3-k folds cross-validations to improve the training performance, and trained for 15 Epochs at each fold.
- For the detection, we used <b>YOLO5</b> which gives more performance.
- All the plots for classification are saved in the sub folder "outputs" which is in the "Classification" folder.
- We could not save the trained model weight for the classification on this github as the file size is 444 MB.
- All the model weight and plots for detection are saved in the sub folder at "Detection\YOLO\yolov5\runs\train\exp" which is in the Detection folder of YOLO.
- The datasets are in the folder "input" for the classification, and in the folder "data" (Detection\YOLO\data) for the detection.
- To train the Detection model use:
  <br/><i> python yolov5/train.py --img 224 --batch 16 --epochs 300 --data dataset.yaml --weights yolov5s.pt</i>
- To Test on an image, use:
  <br/><i>python yolov5/detect.py --weights yolov5/runs/train/exp/weights/best.pt --conf-thres 0.4 --iou-thres 0.75 --source <i>imagelocation</i></i>
  
- You used <b>split.py</b> script to split the training dataset into 80% for training set (1,048 images for Bleeding and 1,048 images for Non-Bleeding) and 20% for validation set (261 images for Bleeding and 261 images for Non-Bleeding). We can see that there is class balance.
- All the CAMs (Class Activation Mapping) Interpretability plots for the images are saved in "Classification\outputs"
  <br />  <br /> 
<b>1• A table of the achieved evaluation metrics of validation dataset</b>
   <b>Classification: Accuracy, Recall, F1-Score </b><br />

               |   Accuracy    |    Recall     |  F1-Score  |
               | ------------- | ------------- | -----------|
               |    99.57%     |    99.53%     |    99.58%  |
          
  <br />
  We also include here a screenshot of the training metrics result.
  <div align="center">
          <a href="figures/metricsClassification.png">
              <img src="figures/metricsClassification.png" width="75%"/>
          </a>
      </div>
  <br />
   <b>Detection: Average Precision, Mean-Average Precision, Intersection over Union(IoU))</b>
   <br />
  
               | Average Precision | Mean-Average Precision | Intersection over Union(IoU)|
               | -------------------| ---------------------- | ----------------------------|
               |         0.70       |         0.682          |             0.45            |


  <br />
  <div align="center">
          <a href="figures/metricsDetection.png">
              <img src="figures/metricsDetection.png" width="100%"/>
          </a>
      </div>
        <br />
<b>2• Screenshots/pictures of any 10 best images selected from validation dataset showing its classification and detection (bounding box with confidence level)</b>
 <br />
    <div align="center">
          <a href="figures/10bestValClassDetection.png">
              <img src="figures/10bestValClassDetection.png" width="75%"/>
          </a>
      </div>
  <br /> 
<b>3• Screenshots/ pictures of achieved interpretability plot of any 10 best images selected from validation dataset</b>
<br/> <b>We use CAMs (Class Activation Mapping) Interpretability in this project</b>
 <br />
         <div align="center">
          <a href="figures/10bestValInterpretabilityPlot.png">
              <img src="figures/10bestValInterpretabilityPlot.png" width="75%"/>
          </a>
         </div>

 <br />
<b>4• Screenshots/pictures of any 5 best images selected from testing dataset 1 and 2 separately showing its classification and detection (bounding box with confidence level)</b> 
<br/> <b>We use CAMs (Class Activation Mapping) Interpretability in this project</b>
<br /> 
         <b>Dataset 1</b>
               <div align="center">
                   <a href="figures/5bestTestDataset1ClassDetection.png">
                       <img src="figures/5bestTestDataset1ClassDetection.png" width="75%"/>
                   </a>
               </div>
              <br /> <br />
           <b>Dataset 2</b>
               <div align="center">
                   <a href="figures/5bestTestDataset2ClassDetection.png">
                       <img src="figures/5bestTestDataset2ClassDetection.png" width="75%"/>
                   </a>
               </div>
            <br /> 
<b>5• Screenshots/ pictures of achieved interpretability plot of any 5 best images selected from testing dataset 1 and 2 separately</b>
           <br /> <b>We use CAMs (Class Activation Mapping) Interpretability in this project</b>
 <br />
         <b>Test Dataset 1</b>
               <div align="center">
                   <a href="figures/5bestTestDataset1InterpretabilityPlot.png">
                       <img src="figures/5bestTestDataset1InterpretabilityPlot.png" width="75%"/>
                   </a>
               </div>
              <br />
           <b>Test Dataset 2</b> 
             <br />
               <div align="center">
                   <a href="figures/5bestTestDataset2InterpretabilityPlot.png">
                       <img src="figures/5bestTestDataset2InterpretabilityPlot.png" width="75%"/>
                   </a>
               </div>
            <br />



<b>6. Excel Sheet Submission:</b>
- The Excel sheet containing the image IDs and predicted class labels of testing dataset 1 and 2 is saved as <b>"Results.xlsx"</b> in the "Detection" folder.
 [click me to download <b>"Results.xlsx"</b>](https://github.com/agossouema2011/WCEBleedGenChallenge/edit/main/Detection/Results.xlsx)
  
