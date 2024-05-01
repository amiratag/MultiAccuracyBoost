# MultiAccuracyBoost
Multiaccuracy: Black-Box Post-Processing for Fairness in Classification

**Please cite the following work if you use this benchmark or the provided tools or implementations:**

```
@inproceedings{kim2019multiaccuracy,
  title={Multiaccuracy: Black-box post-processing for fairness in classification},
  author={Kim, Michael P and Ghorbani, Amirata and Zou, James},
  booktitle={Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society},
  pages={247--254},
  year={2019},
  organization={ACM}
}
```
## Getting Started
Here is the tensorflow implementations of the paper [Multiaccuracy: Black-Box Post-Processing for Fairness in Classification](https://arxiv.org/abs/1805.12317) presented at NeurIPS 2019.

### Prerequisites

Required python libraries:

```
  Scikit-learn: https://scikit-learn.org/stable/
  Tensorflow: https://www.tensorflow.org/
  Facenet: https://github.com/davidsandberg/facenet
```
Also the LFW+A dataset images.

### Installing

Dowanload LFW+A dataset images and put them in a "./LFWA+/lfw" directory. "dataset_description.pkl" maps each image's name to its attributes.

## Authors

* **Michael P. Kim** - [Website](https://cs.stanford.edu/~mpkim/)
* **Amirata Ghorbani** - [Website](http://web.stanford.edu/~amiratag)
* **James Zou** - [Website](https://sites.google.com/site/jamesyzou/)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
