## The MOST exciting and accurate zero-shot classification algorithm from the project! ##

Implementation of the noise-denoise zero-shot classification algorithm. It is so easy and fun to use! That is how:
to use as a CLI-app: put a picture you'd like to classify into the [./to_classify](./to_classify) folder and simply enter this in your command line:

```bash
python CLASSIFY.py
```
## Real-world example. Try it yourself! ##

In all cases diffusion model successfully classified Dan. In the cases of glasses CLIP missclassified him.

| ![dan](pictures/dan.jpg) | ![Sad](pictures/sad.jpg)|![Happy](pictures/happy.jpg) |
|:--:| :--:|:--:|
| *Real Dan* |*A photo of a sad young man*|*A photo of a happy young man*|

| ![dan](pictures/dan.jpg) | ![Sad](pictures/with.jpg)|![Happy](pictures/without.jpg) |
|:--:| :--:|:--:|
| *Real Dan* |*A photo of a young man with glasses*|*A photo of a young man without glasses*|

| ![dan](pictures/dan.jpg) | ![Sad](pictures/white.jpg)|![Happy](pictures/black.jpg) |
|:--:| :--:|:--:|
| *Real Dan* |*A photo of a white young man*|*A photo of a black young man*|
