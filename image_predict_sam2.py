import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import glob
from datetime import datetime

startTime = datetime.now()

## Enviroment setup for pytorch
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def show_img_grid(img1):
    image = Image.open(img1)
    image = np.array(image.convert("RGB"))

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    #show_anns(masks)
    plt.axis('on')
    plt.title(img1[47:-11])
    print(img1)
    plt.show()

##Select pre-trained model pt file
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

## Tiny model
sam2_checkpoint = "/Users/zshane/Documents/sam2/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

## Small model
#sam2_checkpoint = "/Users/zshane/Documents/sam2/sam2/checkpoints/sam2.1_hiera_small.pt"
#model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

## Based+ model
#sam2_checkpoint = "/Users/zshane/Documents/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
#model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

## Large model
#sam2_checkpoint = "/Users/zshane/Documents/sam2/sam2/checkpoints/sam2.1_hiera_large.pt"
#model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

sp_path = '/Users/zshane/Downloads/H. sanguinolenta/*.jpg'

#Create masks folder for species
if not os.path.exists(f"{sp_path[:-6]}/masks0"):
    for i in range(12) :
        os.mkdir(f"{sp_path[:-6]}/masks{i}")

#coordinates for point prompt
arr1 = [[441.9684119492247, 364.91408929670763],
 [543.2847028817683, 355.0638943449326],
 [475.74050892673927, 384.61447920025785],
 [557.3564099557328, 374.7642842484828],
 [617.8647503737799, 355.0638943449326],
 [495.44089883028937, 394.4646741520329],
 [651.6368473512945, 373.3571135410863],
 [802.2041130427136, 362.0997478819148],
 [582.6854826888688, 407.12921051860087],
 [754.3603089912345, 401.50052768901514],
 [916.1849403418254, 414.1650640555831],
 [371.60987657940257, 405.72203981120447],
 [450.4114361936033, 409.9435519333938],
 [575.6496291518865, 432.4582832517368],
 [733.2527483802879, 449.3443317404941],
 [921.813623171411, 469.04472164404433],
 [440.56124124182816, 428.2367711295475],
 [434.93255841224243, 449.3443317404941],
 [624.900603910762, 484.5235994254052],
 [782.5037231391633, 559.1036469174165],
 [910.5562575122395, 533.7745741842806],
 [550.3205564187506, 536.5889155990735],
 [451.8186069009997, 539.4032570138664]]

#seperate points into 3 sets for different segmentation
batch_1pt = np.array([arr1[16:17]+arr1[17:18],
                        arr1[17:18]+arr1[16:17],
                        arr1[18:19]+arr1[21:22],
                        arr1[21:22]+arr1[22:23],
                        arr1[22:23]+arr1[21:22]]) 

batch_2pts = np.array([arr1[0:4],
                        arr1[14:16]+arr1[10:11]+arr1[19:20],
                        arr1[19:21]+arr1[14:16]])

batch_3pts = np.array([arr1[2:5]+arr1[1:2]+arr1[5:6],
                        arr1[5:8]+arr1[3:4]+arr1[9:10],
                        arr1[8:11]+arr1[7:8]+arr1[11:12],
                        arr1[11:14]+arr1[10:11]+arr1[17:18]]) 

#labels - 1 for add masks / 0 for deduct masks
labels_3pts = np.array([[1,1,1,0,0],[1,1,1,0,0],[1,1,1,0,0],[1,1,1,0,0]])
labels_2pts = np.array([[1,1,0,0], [1,1,0,0],[1,1,0,0]])
labels_1pt = np.array([[1,0], [1,0], [1,0], [1,0], [1,0]])

image_paths = glob.glob(sp_path)
for img_path in image_paths:
    image = Image.open(img_path)
    image = np.array(image.convert("RGB"))
    dir_name = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)

    ##Set predictor
    # Predictor for single image
    predictor.set_image(image)

    cnt = 0 #for output file naming 
    '''
    #3 points predictor
    for i in range(len(batch_3pts)) :
        masks, scores, _ = predictor.predict(
            point_coords=batch_3pts[i],
            point_labels=labels_3pts[i],
            multimask_output=False
        )
        
        #show_masks(image, masks, scores, point_coords=batch_3pts[i], input_labels=labels_3pts[i], borders=True)
        cnt = i
        # Convert the mask to a PIL Image
        mask_image = Image.fromarray((masks[0] * 255).astype(np.uint8))
        # Save the image
        mask_image.save(f'{dir_name}/masks{cnt}/{img_name[:-4]}_mask{cnt}.png')
        
        print(f" 3 points mask_{cnt} saved\n")

    print(" 3 Points Prediction done\n")
    '''
    cnt += 1

    #2 points predictor
    for i in range(len(batch_2pts)) :
        masks, scores, _ = predictor.predict(
            point_coords=batch_2pts[i],
            point_labels=labels_2pts[i],
            multimask_output=False
        )
        #print(batch_2pts[i])
        #print(labels_2pts[i])

        #show_masks(image, masks, scores, point_coords=batch_2pts[i], input_labels=labels_2pts[i], borders=True)
        # Convert the mask to a PIL Image
        mask_image = Image.fromarray((masks[0] * 255).astype(np.uint8))
        # Save the image
        mask_image.save(f'{dir_name}/masks{cnt}/{img_name[:-4]}_mask{cnt}.png')
        print(f" 2 points Mask {cnt} saved successfully\n")
        cnt += 1

    print(" 2 Points Prediction done\n")

    #1 point predictor
    for i in range(len(batch_1pt)) :
        masks, scores, _ = predictor.predict(
            point_coords=batch_1pt[i],
            point_labels=labels_1pt[i],
            multimask_output=False
        )
        mask_image = Image.fromarray((masks[0] * 255).astype(np.uint8))
        # Save the image
        mask_image.save(f'{dir_name}/masks{cnt}/{img_name[:-4]}_mask{cnt}.png')
        print(f" 1 point Mask {cnt} saved successfully.\n")
        cnt += 1

    print(" 1 Points Prediction done")
    print(f"{img_name} SAM masks generation done\n")

print(f"Total runtime for {sp_path}: {datetime.now() - startTime} minutes")  # Time taken in minutes