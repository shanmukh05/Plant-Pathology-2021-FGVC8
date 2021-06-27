# Plant-Pathology-2021-FGVC8
### Identify the category of foliar diseases in apple trees

<h1>Details of the Project</h1>
<p> Plant Pathology competition is a annual competition conducted by <b>FGVC8</b> to detect foilar diseases in Apple Plants. </p>

[Check out the competition here](https://www.kaggle.com/c/plant-pathology-2021-fgvc8/overview/description)

- Host : **Kaggle**
- Dataset : **23,000 high-quality RGB images of apple foliar diseases**
- Evaluation Metric : **Mean F1-score**
- Framework used : **TensorFlow**
- Accelerators : **GPU**, **TPU**

# Important information about labels
This is a multi-label task i.e., an image can have more than one label.
- Number of classes : 6 (`scab`, `healthy`, `frog eye leaf spot`, `rust`, `complex`, `powdery mildew`)
- Number of unique labels : 12 (above 6 + `scab, frog eye leaf spot`, `scab frog eye leaf spot complex`, `frog eye leaf spot, complex`, `rust frog eye leaf spot`, `rust complex`, `powdery mildew, complex`)

# Approach to the Competition
I approached the task in 3 ways
- Considering number of unique labels as 12 (assuming no other combination of diseases) and proceded as Multi Class classification
- Considering number of unique labels as 6 (Multi label classification).
- Considering number of unique labels as 5 (if no label in 5 gave good probability then it is `healthy`).
- Last approach gave best resluts.

# Other Details 
- ## Dataset Preparation
  - TPU as Accelerator
     - As the dataset is present in GCP, we can't access the images directly, We first need to decode the image using `image = tf.image.decode_jpeg(tf.io.read_file(filepath), channels=3)`
     - After loading the images, random augmentations are applied using `tf.image.random.` class.
     - Finally to prepare the datset for training, loaded images are converted to Tensorflow `tf.data.Dataset` format and adding `cache`, `prefetch` for faster loading of data. 
     - [More details are in TPU notebook]().
  - GPU as Accelerator
     - `ImageDataGenerator` is used to apply several augmentations to images.
     - Images are loaded as datset using `flow_from_dataframe` function.
     - [More details in GPU notebook]().

- ## Training 
  - In both cases (TPU,GPU) pretrained models present in `tf.keras.applications` are used.
  - Loss : Binary Cross Entropy.
  - Metrics tracked throughput training are `F1-score` , `Accuracy`.
  - Callbacks used
      - `ModelCheckpoint` - to store the best model.
      - `ReduceRONPlateau` - to change learning rate if there is no improvement observed.
      - `EarlyStopping` - to stop training i case of no improvement.
  - TPU as Accelerator
      - As we can't use any tensorflow dataset loaders, `kfold split` of `sklearn` is used to split the data into `training` and `validation`.    
  - GPU as Accelerator
     - using `split` option in `ImageDataGenerator` datset is splitted into `training` and `validation`.

- ## Submission
  - Finally the best models are loaded and used to predict new labels.
  - [More details in Inference Notebook]().
  - [Sample submission csv file]() can be found here.
