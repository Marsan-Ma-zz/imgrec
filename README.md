# tflearn (tensorflow-learn) GoogleNet image recognition framework


this is a image recognition framework, based on tflearn (tensorflow-learn) and GoogLeNet image recognition model.

## How To Use

1. Arrange your own data set in `images/<DATASET_NAME>/<LABEL_NAME>`
2. For example, you could do `python3 dump_17flowers.py` to download  the example dataset [17 Category Flower Dataset](a2) from Oxford university. The data hierarchy see the folders hierarchy like this:
   
   ```
    images  
      |---17flowers  
             |--- 0  
             |--- 1  
             |--- 2  
             ...
             |--- 15  
             |--- 16  
             
    Each folder (0 to 16) contains samples of one label in dataset.
   ```

3. Then just type `python3 imgrec.py` to train the model, the latest models will be saved in `models/<DATASET_NAME>`. If training process is interrupt, it will find and restore the latest trained model from this folder.
   
4. After you are satisfied with the training result, you could start the Flask server by `python3 app.py`, fill in the new picture you want to classify and submit, you will get prediction result like this:

![webapp.png](webapp.png)


[a1]: https://github.com/tflearn/tflearn/blob/master/examples/images/googlenet.py
[a2]: http://www.robots.ox.ac.uk/~vgg/data/flowers/17/




## Params

If you have more than one project to share your GPU memory, you could reduce the GPU memory usage by changing the `gpu_memory_fraction` in `imgrec.py` to, like 0.5 for 2 projects or 0.3 for 3 projects to share your computing resource.


