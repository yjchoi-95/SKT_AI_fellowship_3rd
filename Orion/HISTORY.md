History
=======

## 0.1.7 - 2021-05-04

This version adds new features to the benchmark function where users can now save pipelines, view results as they are being calculated, and allow a single evaluation to be compared multiple times.

### Issues resolved
* Dask issues in benchmark function & improvements - [Issue #225](https://github.com/signals-dev/Orion/issues/225) by @sarahmish
* Numerical overflow when using contextual metrics - [Issue #212](https://github.com/signals-dev/Orion/issues/212) by @kronerte


## 0.1.6 - 2021-03-08

This version introduces two new pipelines: LSTM AE and Dense AE.
In addition to minor improvements, a bit of code refactoring took place to introduce
a new primtive: ``reconstruction_errors``.

### Issues resolved
* Comparison of DTW library performance - [Issue #205](https://github.com/signals-dev/Orion/issues/205) by @sarahmish
* Not able to pickle dump tadgan pipeline - [Issue #200](https://github.com/signals-dev/Orion/issues/200) by @sarahmish
* New pipeline LSTM and Dense autoencoders - [Issue #194](https://github.com/signals-dev/Orion/issues/194) by @sarahmish
* Readme - [Issue #192](https://github.com/signals-dev/Orion/issues/192) by @pvk-developer
* Unable to launch cli - [Issue #186](https://github.com/signals-dev/Orion/issues/186) by @sarahmish
* bullet points not formatted correctly in index.rst - [Issue #178](https://github.com/signals-dev/Orion/issues/178) by @micahjsmith
* Update notebooks - [Issue #176](https://github.com/signals-dev/Orion/issues/176) by @sarahmish
* Inaccuracy in README.md file in orion/evaluation/ - [Issue #157](https://github.com/signals-dev/Orion/issues/157) by @sarahmish
* Dockerfile -- docker does not find orion primitives automatically - [Issue #155](https://github.com/signals-dev/Orion/issues/155) by @sarahmish
* Primitive documentation - [Issue #151](https://github.com/signals-dev/Orion/issues/151) by @sarahmish
* Variable name inconsistency in tadgan - [Issue #150](https://github.com/signals-dev/Orion/issues/150) by @sarahmish
* Sync leaderboard tables between `BENCHMARK.md` and the docs - [Issue #148](https://github.com/signals-dev/Orion/issues/148) by @sarahmish


## 0.1.5 - 2020-12-25

This version includes the new style of documentation and a revamp of the `README.md`. In addition to some minor improvements
in the benchmark code and primitives. This release includes the transfer of `tadgan` pipeline to `verified`.

### Issues resolved
* Link with google colab - [Issue #144](https://github.com/signals-dev/Orion/issues/144) by @sarahmish
* Add `timeseries_anomalies` unittests - [Issue #136](https://github.com/signals-dev/Orion/issues/136) by @sarahmish
* Update `find_sequences` in converting series to arrays - [Issue #135](https://github.com/signals-dev/Orion/issues/135) by @sarahmish
* Definition of error/critic smooth window in score anomalies primitive - [Issue #132](https://github.com/signals-dev/Orion/issues/132) by @sarahmish
* Train-test split in benchmark enhancement - [Issue #130](https://github.com/signals-dev/Orion/issues/130) by @sarahmish


## 0.1.4 - 2020-10-16

Minor enhancements to benchmark

* Load ground truth before try-catch - [Issue #124](https://github.com/signals-dev/Orion/issues/124) by @sarahmish
* Converting timestamp to datetime in Azure primitive - [Issue #123](https://github.com/signals-dev/Orion/issues/123) by @sarahmish
* Benchmark exceptions - [Issue #120](https://github.com/signals-dev/Orion/issues/120) by @sarahmish


## 0.1.3 - 2020-09-29

New benchmark and Azure primitive.

* Implement a benchmarking function new feature - [Issue #94](https://github.com/signals-dev/Orion/issues/94) by @sarahmish
* Add azure anomaly detection as primitive new feature - [Issue #97](https://github.com/signals-dev/Orion/issues/97) by @sarahmish
* Critic and reconstruction error combination - [Issue #99](https://github.com/signals-dev/Orion/issues/99) by @sarahmish
* Fixed threshold for `find_anomalies` - [Issue #101](https://github.com/signals-dev/Orion/issues/101) by @sarahmish
* Add an option to have window size and window step size as percentages of error size - [Issue #102](https://github.com/signals-dev/Orion/issues/102) by @sarahmish
* Organize pipelines into verified and sandbox - [Issue #105](https://github.com/signals-dev/Orion/issues/105) by @sarahmish
* Ground truth parameter name enhancement - [Issue #114](https://github.com/signals-dev/Orion/issues/114) by @sarahmish
* Add benchmark dataset list and parameters to s3 bucket enhancement - [Issue #118](https://github.com/signals-dev/Orion/issues/118) by @sarahmish

## 0.1.2 - 2020-07-03

New Evaluation sub-package and refactor TadGAN.

* Two bugs when saving signalrun if there is no event detected - [Issue #92](https://github.com/signals-dev/Orion/issues/92) by @dyuliu 
* File encoding/decoding issues about `README.md` and `HISTORY.md` - [Issue #88](https://github.com/signals-dev/Orion/issues/88) by @dyuliu
* Fix bottle neck of `score_anomaly` in Cyclegan primitive - [Issue #86](https://github.com/signals-dev/Orion/issues/86) by @dyuliu
* Adjust `epoch` meaning in Cyclegan primitive - [Issue #85](https://github.com/signals-dev/Orion/issues/85) by @sarahmish
* Rename evaluation to benchmark and metrics to evaluation - [Issue #83](https://github.com/signals-dev/Orion/issues/83) by @sarahmish
* Scoring function for intervals of size one - [Issue #76](https://github.com/signals-dev/Orion/issues/76) by @sarahmish

## 0.1.1 - 2020-05-11

New class and function based interfaces.

* Implement the Orion Class - [Issue #79](https://github.com/D3-AI/Orion/issues/79) by @csala
* Implement new functional interface - [Issue #80](https://github.com/D3-AI/Orion/issues/80) by @csala

## 0.1.0 - 2020-04-23

First Orion release to PyPI: https://pypi.org/project/orion-ml/
