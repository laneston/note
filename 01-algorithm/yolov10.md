
```
.
|-- .git
|   |-- FETCH_HEAD
|   |-- HEAD
|   |-- config
|   |-- description
|   |-- hooks
|   |   |-- applypatch-msg.sample
|   |   |-- commit-msg.sample
|   |   |-- fsmonitor-watchman.sample
|   |   |-- post-update.sample
|   |   |-- pre-applypatch.sample
|   |   |-- pre-commit.sample
|   |   |-- pre-merge-commit.sample
|   |   |-- pre-push.sample
|   |   |-- pre-rebase.sample
|   |   |-- pre-receive.sample
|   |   |-- prepare-commit-msg.sample
|   |   |-- push-to-checkout.sample
|   |   |-- sendemail-validate.sample
|   |   `-- update.sample
|   |-- index
|   |-- info
|   |   `-- exclude
|   |-- logs
|   |   |-- HEAD
|   |   `-- refs
|   |       |-- heads
|   |       |   `-- main
|   |       `-- remotes
|   |           |-- origin
|   |           |   `-- HEAD
|   |           `-- upstream
|   |               |-- HEAD
|   |               `-- main
|   |-- objects
|   |   |-- 02
|   |   |   `-- 8699065504950a4d4ae04291da1a1cb7f79de4
|   |   |-- 08
|   |   |   `-- c0b09b13e1d7895864ed5937e0b5086ac7896f
|   |   |-- 0e
|   |   |   `-- 51495ce630ad32b36dd8990797dda6a4a8f68d
|   |   |-- 19
|   |   |   `-- 4d22daadd6a04c3103cc29cc6f4a9490ccb2ca
|   |   |-- 1b
|   |   |   |-- 0827ad351b29a715bb4dd36a0b76ff74b2c468
|   |   |   `-- 2291e8534b4030f0e9ba82b3252e5cc88a26ca
|   |   |-- 28
|   |   |   `-- 480e2d1e46fd096acbf311ec966cbc0f8ef9b1
|   |   |-- 33
|   |   |   `-- d5d5469c7d179cb383861bc7bfa0f9713ecfcf
|   |   |-- 36
|   |   |   |-- 09344b522f34abba76324695c24f3a2038ea69
|   |   |   `-- 2686a7e4de042603183538325cbe8ccaa83c9b
|   |   |-- 46
|   |   |   `-- 26a78d8a1202791f424903045dfaf9cf340436
|   |   |-- 51
|   |   |   `-- 584e38012c5c332af2d767b52b0c56ade68526
|   |   |-- 58
|   |   |   `-- bcbdde3e55365c009916acd0a37e309591211f
|   |   |-- 5f
|   |   |   `-- 5732b27405b35c91564747cc727a44d38b8460
|   |   |-- 6a
|   |   |   `-- c170c55b852c79456bb317993e1646dce581fa
|   |   |-- 74
|   |   |   `-- af702fcd016b325600952f569c31fe41ccdcea
|   |   |-- 7f
|   |   |   |-- 4c4fedb5bf466b1478f0d5f211303a835261fd
|   |   |   `-- dafd9add4927792ec8cfaaa187688e0a8da17b
|   |   |-- 88
|   |   |   `-- 908b294dc28277391e0bb5b335f0979eacb363
|   |   |-- 8b
|   |   |   `-- 883150f9d7a1d7351414041dc2fc775778d08e
|   |   |-- bd
|   |   |   |-- 074b10cc7cdfe6b65dd2032a1762ff1bc0c591
|   |   |   `-- e3a15d2fd56db4bef6dc8e9fbb5af23377b1a9
|   |   |-- cd
|   |   |   `-- 2f79c70299c9041fb6d19617ef1296f47575b1
|   |   |-- f4
|   |   |   `-- e8a63dea7acdc787205e2ab965deab7cb7ced0
|   |   |-- f6
|   |   |   `-- ef8b45be3274579982e6291bb2e796b81f7f0b
|   |   |-- f9
|   |   |   `-- d5ac764a85d526209bb9933f0b328e20ff6e46
|   |   |-- info
|   |   `-- pack
|   |       |-- pack-4856341ea217eefbf0cb672bd22606bdcd9683ed.idx
|   |       |-- pack-4856341ea217eefbf0cb672bd22606bdcd9683ed.pack
|   |       `-- pack-4856341ea217eefbf0cb672bd22606bdcd9683ed.rev
|   |-- packed-refs
|   `-- refs
|       |-- heads
|       |   `-- main
|       |-- remotes
|       |   |-- origin
|       |   |   `-- HEAD
|       |   `-- upstream
|       |       |-- HEAD
|       |       `-- main
|       `-- tags
|           |-- v1.0
|           `-- v1.1
|-- .gitignore
|-- .pre-commit-config.yaml
|-- CONTRIBUTING.md
|-- LICENSE
|-- README.md
|-- app.py
|-- docker
|   |-- Dockerfile
|   |-- Dockerfile-arm64
|   |-- Dockerfile-conda
|   |-- Dockerfile-cpu
|   |-- Dockerfile-jetson
|   |-- Dockerfile-python
|   `-- Dockerfile-runner
|-- docs
|   |-- README.md
|   |-- build_docs.py
|   |-- build_reference.py
|   |-- coming_soon_template.md
|   |-- en
|   |   |-- CNAME
|   |   |-- datasets
|   |   |   |-- classify
|   |   |   |   |-- caltech101.md
|   |   |   |   |-- caltech256.md
|   |   |   |   |-- cifar10.md
|   |   |   |   |-- cifar100.md
|   |   |   |   |-- fashion-mnist.md
|   |   |   |   |-- imagenet.md
|   |   |   |   |-- imagenet10.md
|   |   |   |   |-- imagenette.md
|   |   |   |   |-- imagewoof.md
|   |   |   |   |-- index.md
|   |   |   |   `-- mnist.md
|   |   |   |-- detect
|   |   |   |   |-- african-wildlife.md
|   |   |   |   |-- argoverse.md
|   |   |   |   |-- brain-tumor.md
|   |   |   |   |-- coco.md
|   |   |   |   |-- coco8.md
|   |   |   |   |-- globalwheat2020.md
|   |   |   |   |-- index.md
|   |   |   |   |-- objects365.md
|   |   |   |   |-- open-images-v7.md
|   |   |   |   |-- roboflow-100.md
|   |   |   |   |-- sku-110k.md
|   |   |   |   |-- visdrone.md
|   |   |   |   |-- voc.md
|   |   |   |   `-- xview.md
|   |   |   |-- explorer
|   |   |   |   |-- api.md
|   |   |   |   |-- dashboard.md
|   |   |   |   |-- explorer.ipynb
|   |   |   |   `-- index.md
|   |   |   |-- index.md
|   |   |   |-- obb
|   |   |   |   |-- dota-v2.md
|   |   |   |   |-- dota8.md
|   |   |   |   `-- index.md
|   |   |   |-- pose
|   |   |   |   |-- coco.md
|   |   |   |   |-- coco8-pose.md
|   |   |   |   |-- index.md
|   |   |   |   `-- tiger-pose.md
|   |   |   |-- segment
|   |   |   |   |-- carparts-seg.md
|   |   |   |   |-- coco.md
|   |   |   |   |-- coco8-seg.md
|   |   |   |   |-- crack-seg.md
|   |   |   |   |-- index.md
|   |   |   |   `-- package-seg.md
|   |   |   `-- track
|   |   |       `-- index.md
|   |   |-- guides
|   |   |   |-- azureml-quickstart.md
|   |   |   |-- conda-quickstart.md
|   |   |   |-- coral-edge-tpu-on-raspberry-pi.md
|   |   |   |-- distance-calculation.md
|   |   |   |-- docker-quickstart.md
|   |   |   |-- heatmaps.md
|   |   |   |-- hyperparameter-tuning.md
|   |   |   |-- index.md
|   |   |   |-- instance-segmentation-and-tracking.md
|   |   |   |-- isolating-segmentation-objects.md
|   |   |   |-- kfold-cross-validation.md
|   |   |   |-- model-deployment-options.md
|   |   |   |-- object-blurring.md
|   |   |   |-- object-counting.md
|   |   |   |-- object-cropping.md
|   |   |   |-- optimizing-openvino-latency-vs-throughput-modes.md
|   |   |   |-- raspberry-pi.md
|   |   |   |-- region-counting.md
|   |   |   |-- sahi-tiled-inference.md
|   |   |   |-- security-alarm-system.md
|   |   |   |-- speed-estimation.md
|   |   |   |-- triton-inference-server.md
|   |   |   |-- view-results-in-terminal.md
|   |   |   |-- vision-eye.md
|   |   |   |-- workouts-monitoring.md
|   |   |   |-- yolo-common-issues.md
|   |   |   |-- yolo-performance-metrics.md
|   |   |   `-- yolo-thread-safe-inference.md
|   |   |-- help
|   |   |   |-- CI.md
|   |   |   |-- CLA.md
|   |   |   |-- FAQ.md
|   |   |   |-- code_of_conduct.md
|   |   |   |-- contributing.md
|   |   |   |-- environmental-health-safety.md
|   |   |   |-- index.md
|   |   |   |-- minimum_reproducible_example.md
|   |   |   |-- privacy.md
|   |   |   `-- security.md
|   |   |-- hub
|   |   |   |-- api
|   |   |   |   `-- index.md
|   |   |   |-- app
|   |   |   |   |-- android.md
|   |   |   |   |-- index.md
|   |   |   |   `-- ios.md
|   |   |   |-- cloud-training.md
|   |   |   |-- datasets.md
|   |   |   |-- index.md
|   |   |   |-- inference-api.md
|   |   |   |-- integrations.md
|   |   |   |-- models.md
|   |   |   |-- on-premise
|   |   |   |   `-- index.md
|   |   |   |-- projects.md
|   |   |   `-- quickstart.md
|   |   |-- index.md
|   |   |-- integrations
|   |   |   |-- amazon-sagemaker.md
|   |   |   |-- clearml.md
|   |   |   |-- comet.md
|   |   |   |-- coreml.md
|   |   |   |-- dvc.md
|   |   |   |-- edge-tpu.md
|   |   |   |-- gradio.md
|   |   |   |-- index.md
|   |   |   |-- mlflow.md
|   |   |   |-- ncnn.md
|   |   |   |-- neural-magic.md
|   |   |   |-- onnx.md
|   |   |   |-- openvino.md
|   |   |   |-- paddlepaddle.md
|   |   |   |-- ray-tune.md
|   |   |   |-- roboflow.md
|   |   |   |-- tensorboard.md
|   |   |   |-- tensorrt.md
|   |   |   |-- tf-graphdef.md
|   |   |   |-- tf-savedmodel.md
|   |   |   |-- tflite.md
|   |   |   |-- torchscript.md
|   |   |   `-- weights-biases.md
|   |   |-- models
|   |   |   |-- fast-sam.md
|   |   |   |-- index.md
|   |   |   |-- mobile-sam.md
|   |   |   |-- rtdetr.md
|   |   |   |-- sam.md
|   |   |   |-- yolo-nas.md
|   |   |   |-- yolo-world.md
|   |   |   |-- yolov3.md
|   |   |   |-- yolov4.md
|   |   |   |-- yolov5.md
|   |   |   |-- yolov6.md
|   |   |   |-- yolov7.md
|   |   |   |-- yolov8.md
|   |   |   `-- yolov9.md
|   |   |-- modes
|   |   |   |-- benchmark.md
|   |   |   |-- export.md
|   |   |   |-- index.md
|   |   |   |-- predict.md
|   |   |   |-- track.md
|   |   |   |-- train.md
|   |   |   `-- val.md
|   |   |-- quickstart.md
|   |   |-- reference
|   |   |   |-- cfg
|   |   |   |   `-- __init__.md
|   |   |   |-- data
|   |   |   |   |-- annotator.md
|   |   |   |   |-- augment.md
|   |   |   |   |-- base.md
|   |   |   |   |-- build.md
|   |   |   |   |-- converter.md
|   |   |   |   |-- dataset.md
|   |   |   |   |-- explorer
|   |   |   |   |   |-- explorer.md
|   |   |   |   |   |-- gui
|   |   |   |   |   |   `-- dash.md
|   |   |   |   |   `-- utils.md
|   |   |   |   |-- loaders.md
|   |   |   |   |-- split_dota.md
|   |   |   |   `-- utils.md
|   |   |   |-- engine
|   |   |   |   |-- exporter.md
|   |   |   |   |-- model.md
|   |   |   |   |-- predictor.md
|   |   |   |   |-- results.md
|   |   |   |   |-- trainer.md
|   |   |   |   |-- tuner.md
|   |   |   |   `-- validator.md
|   |   |   |-- hub
|   |   |   |   |-- __init__.md
|   |   |   |   |-- auth.md
|   |   |   |   |-- session.md
|   |   |   |   `-- utils.md
|   |   |   |-- models
|   |   |   |   |-- fastsam
|   |   |   |   |   |-- model.md
|   |   |   |   |   |-- predict.md
|   |   |   |   |   |-- prompt.md
|   |   |   |   |   |-- utils.md
|   |   |   |   |   `-- val.md
|   |   |   |   |-- nas
|   |   |   |   |   |-- model.md
|   |   |   |   |   |-- predict.md
|   |   |   |   |   `-- val.md
|   |   |   |   |-- rtdetr
|   |   |   |   |   |-- model.md
|   |   |   |   |   |-- predict.md
|   |   |   |   |   |-- train.md
|   |   |   |   |   `-- val.md
|   |   |   |   |-- sam
|   |   |   |   |   |-- amg.md
|   |   |   |   |   |-- build.md
|   |   |   |   |   |-- model.md
|   |   |   |   |   |-- modules
|   |   |   |   |   |   |-- decoders.md
|   |   |   |   |   |   |-- encoders.md
|   |   |   |   |   |   |-- sam.md
|   |   |   |   |   |   |-- tiny_encoder.md
|   |   |   |   |   |   `-- transformer.md
|   |   |   |   |   `-- predict.md
|   |   |   |   |-- utils
|   |   |   |   |   |-- loss.md
|   |   |   |   |   `-- ops.md
|   |   |   |   `-- yolo
|   |   |   |       |-- classify
|   |   |   |       |   |-- predict.md
|   |   |   |       |   |-- train.md
|   |   |   |       |   `-- val.md
|   |   |   |       |-- detect
|   |   |   |       |   |-- predict.md
|   |   |   |       |   |-- train.md
|   |   |   |       |   `-- val.md
|   |   |   |       |-- model.md
|   |   |   |       |-- obb
|   |   |   |       |   |-- predict.md
|   |   |   |       |   |-- train.md
|   |   |   |       |   `-- val.md
|   |   |   |       |-- pose
|   |   |   |       |   |-- predict.md
|   |   |   |       |   |-- train.md
|   |   |   |       |   `-- val.md
|   |   |   |       `-- segment
|   |   |   |           |-- predict.md
|   |   |   |           |-- train.md
|   |   |   |           `-- val.md
|   |   |   |-- nn
|   |   |   |   |-- autobackend.md
|   |   |   |   |-- modules
|   |   |   |   |   |-- block.md
|   |   |   |   |   |-- conv.md
|   |   |   |   |   |-- head.md
|   |   |   |   |   |-- transformer.md
|   |   |   |   |   `-- utils.md
|   |   |   |   `-- tasks.md
|   |   |   |-- solutions
|   |   |   |   |-- ai_gym.md
|   |   |   |   |-- distance_calculation.md
|   |   |   |   |-- heatmap.md
|   |   |   |   |-- object_counter.md
|   |   |   |   `-- speed_estimation.md
|   |   |   |-- trackers
|   |   |   |   |-- basetrack.md
|   |   |   |   |-- bot_sort.md
|   |   |   |   |-- byte_tracker.md
|   |   |   |   |-- track.md
|   |   |   |   `-- utils
|   |   |   |       |-- gmc.md
|   |   |   |       |-- kalman_filter.md
|   |   |   |       `-- matching.md
|   |   |   `-- utils
|   |   |       |-- __init__.md
|   |   |       |-- autobatch.md
|   |   |       |-- benchmarks.md
|   |   |       |-- callbacks
|   |   |       |   |-- base.md
|   |   |       |   |-- clearml.md
|   |   |       |   |-- comet.md
|   |   |       |   |-- dvc.md
|   |   |       |   |-- hub.md
|   |   |       |   |-- mlflow.md
|   |   |       |   |-- neptune.md
|   |   |       |   |-- raytune.md
|   |   |       |   |-- tensorboard.md
|   |   |       |   `-- wb.md
|   |   |       |-- checks.md
|   |   |       |-- dist.md
|   |   |       |-- downloads.md
|   |   |       |-- errors.md
|   |   |       |-- files.md
|   |   |       |-- instance.md
|   |   |       |-- loss.md
|   |   |       |-- metrics.md
|   |   |       |-- ops.md
|   |   |       |-- patches.md
|   |   |       |-- plotting.md
|   |   |       |-- tal.md
|   |   |       |-- torch_utils.md
|   |   |       |-- triton.md
|   |   |       `-- tuner.md
|   |   |-- robots.txt
|   |   |-- tasks
|   |   |   |-- classify.md
|   |   |   |-- detect.md
|   |   |   |-- index.md
|   |   |   |-- obb.md
|   |   |   |-- pose.md
|   |   |   `-- segment.md
|   |   |-- usage
|   |   |   |-- callbacks.md
|   |   |   |-- cfg.md
|   |   |   |-- cli.md
|   |   |   |-- engine.md
|   |   |   |-- python.md
|   |   |   `-- simple-utilities.md
|   |   `-- yolov5
|   |       |-- environments
|   |       |   |-- aws_quickstart_tutorial.md
|   |       |   |-- azureml_quickstart_tutorial.md
|   |       |   |-- docker_image_quickstart_tutorial.md
|   |       |   `-- google_cloud_quickstart_tutorial.md
|   |       |-- index.md
|   |       |-- quickstart_tutorial.md
|   |       `-- tutorials
|   |           |-- architecture_description.md
|   |           |-- clearml_logging_integration.md
|   |           |-- comet_logging_integration.md
|   |           |-- hyperparameter_evolution.md
|   |           |-- model_ensembling.md
|   |           |-- model_export.md
|   |           |-- model_pruning_and_sparsity.md
|   |           |-- multi_gpu_training.md
|   |           |-- neural_magic_pruning_quantization.md
|   |           |-- pytorch_hub_model_loading.md
|   |           |-- roboflow_datasets_integration.md
|   |           |-- running_on_jetson_nano.md
|   |           |-- test_time_augmentation.md
|   |           |-- tips_for_best_training_results.md
|   |           |-- train_custom_data.md
|   |           `-- transfer_learning_with_frozen_layers.md
|   |-- mkdocs_github_authors.yaml
|   `-- overrides
|       |-- assets
|       |   `-- favicon.ico
|       |-- javascript
|       |   `-- extra.js
|       |-- main.html
|       |-- partials
|       |   |-- comments.html
|       |   `-- source-file.html
|       `-- stylesheets
|           `-- style.css
|-- examples
|   |-- README.md
|   |-- YOLOv8-CPP-Inference
|   |   |-- CMakeLists.txt
|   |   |-- README.md
|   |   |-- inference.cpp
|   |   |-- inference.h
|   |   `-- main.cpp
|   |-- YOLOv8-LibTorch-CPP-Inference
|   |   |-- CMakeLists.txt
|   |   |-- README.md
|   |   `-- main.cc
|   |-- YOLOv8-ONNXRuntime
|   |   |-- README.md
|   |   `-- main.py
|   |-- YOLOv8-ONNXRuntime-CPP
|   |   |-- CMakeLists.txt
|   |   |-- README.md
|   |   |-- inference.cpp
|   |   |-- inference.h
|   |   `-- main.cpp
|   |-- YOLOv8-ONNXRuntime-Rust
|   |   |-- Cargo.toml
|   |   |-- README.md
|   |   `-- src
|   |       |-- cli.rs
|   |       |-- lib.rs
|   |       |-- main.rs
|   |       |-- model.rs
|   |       |-- ort_backend.rs
|   |       `-- yolo_result.rs
|   |-- YOLOv8-OpenCV-ONNX-Python
|   |   |-- README.md
|   |   `-- main.py
|   |-- YOLOv8-OpenCV-int8-tflite-Python
|   |   |-- README.md
|   |   `-- main.py
|   |-- YOLOv8-Region-Counter
|   |   |-- readme.md
|   |   `-- yolov8_region_counter.py
|   |-- YOLOv8-SAHI-Inference-Video
|   |   |-- readme.md
|   |   `-- yolov8_sahi.py
|   |-- YOLOv8-Segmentation-ONNXRuntime-Python
|   |   |-- README.md
|   |   `-- main.py
|   |-- heatmaps.ipynb
|   |-- hub.ipynb
|   |-- object_counting.ipynb
|   |-- object_tracking.ipynb
|   `-- tutorial.ipynb
|-- figures
|   |-- latency.svg
|   `-- params.svg
|-- flops.py
|-- gradio_cached_examples
|   `-- 19
|       |-- Annotated Image
|       |   |-- 9ac77dc07976339515bf
|       |   |   `-- image.webp
|       |   `-- da3a135888a1ef24b4af
|       |       `-- image.webp
|       |-- indices.csv
|       `-- log.csv
|-- logs
|   |-- yolov10b.csv
|   |-- yolov10l.csv
|   |-- yolov10m.csv
|   |-- yolov10n.csv
|   |-- yolov10s.csv
|   `-- yolov10x.csv
|-- mkdocs.yml
|-- pyproject.toml
|-- requirements.txt
|-- tests
|   |-- conftest.py
|   |-- test_cli.py
|   |-- test_cuda.py
|   |-- test_engine.py
|   |-- test_explorer.py
|   |-- test_integrations.py
|   `-- test_python.py
|-- ultralytics
|   |-- __init__.py
|   |-- __pycache__
|   |   `-- __init__.cpython-39.pyc
|   |-- assets
|   |   |-- bus.jpg
|   |   `-- zidane.jpg
|   |-- cfg
|   |   |-- __init__.py
|   |   |-- __pycache__
|   |   |   `-- __init__.cpython-39.pyc
|   |   |-- datasets
|   |   |   |-- Argoverse.yaml
|   |   |   |-- DOTAv1.5.yaml
|   |   |   |-- DOTAv1.yaml
|   |   |   |-- GlobalWheat2020.yaml
|   |   |   |-- ImageNet.yaml
|   |   |   |-- Objects365.yaml
|   |   |   |-- SKU-110K.yaml
|   |   |   |-- VOC.yaml
|   |   |   |-- VisDrone.yaml
|   |   |   |-- african-wildlife.yaml
|   |   |   |-- brain-tumor.yaml
|   |   |   |-- carparts-seg.yaml
|   |   |   |-- coco-pose.yaml
|   |   |   |-- coco.yaml
|   |   |   |-- coco128-seg.yaml
|   |   |   |-- coco128.yaml
|   |   |   |-- coco8-pose.yaml
|   |   |   |-- coco8-seg.yaml
|   |   |   |-- coco8.yaml
|   |   |   |-- crack-seg.yaml
|   |   |   |-- dota8.yaml
|   |   |   |-- open-images-v7.yaml
|   |   |   |-- package-seg.yaml
|   |   |   |-- tiger-pose.yaml
|   |   |   `-- xView.yaml
|   |   |-- default.yaml
|   |   |-- models
|   |   |   |-- README.md
|   |   |   |-- rt-detr
|   |   |   |   |-- rtdetr-l.yaml
|   |   |   |   |-- rtdetr-resnet101.yaml
|   |   |   |   |-- rtdetr-resnet50.yaml
|   |   |   |   `-- rtdetr-x.yaml
|   |   |   |-- v10
|   |   |   |   |-- yolov10b.yaml
|   |   |   |   |-- yolov10l.yaml
|   |   |   |   |-- yolov10m.yaml
|   |   |   |   |-- yolov10n.yaml
|   |   |   |   |-- yolov10s.yaml
|   |   |   |   `-- yolov10s.yaml.yaml
|   |   |   |-- v3
|   |   |   |   |-- yolov3-spp.yaml
|   |   |   |   |-- yolov3-tiny.yaml
|   |   |   |   `-- yolov3.yaml
|   |   |   |-- v5
|   |   |   |   |-- yolov5-p6.yaml
|   |   |   |   `-- yolov5.yaml
|   |   |   |-- v6
|   |   |   |   `-- yolov6.yaml
|   |   |   |-- v8
|   |   |   |   |-- yolov8-cls-resnet101.yaml
|   |   |   |   |-- yolov8-cls-resnet50.yaml
|   |   |   |   |-- yolov8-cls.yaml
|   |   |   |   |-- yolov8-ghost-p2.yaml
|   |   |   |   |-- yolov8-ghost-p6.yaml
|   |   |   |   |-- yolov8-ghost.yaml
|   |   |   |   |-- yolov8-obb.yaml
|   |   |   |   |-- yolov8-p2.yaml
|   |   |   |   |-- yolov8-p6.yaml
|   |   |   |   |-- yolov8-pose-p6.yaml
|   |   |   |   |-- yolov8-pose.yaml
|   |   |   |   |-- yolov8-rtdetr.yaml
|   |   |   |   |-- yolov8-seg-p6.yaml
|   |   |   |   |-- yolov8-seg.yaml
|   |   |   |   |-- yolov8-world.yaml
|   |   |   |   |-- yolov8-worldv2.yaml
|   |   |   |   `-- yolov8.yaml
|   |   |   `-- v9
|   |   |       |-- yolov9c.yaml
|   |   |       `-- yolov9e.yaml
|   |   `-- trackers
|   |       |-- botsort.yaml
|   |       `-- bytetrack.yaml
|   |-- data
|   |   |-- __init__.py
|   |   |-- __pycache__
|   |   |   |-- __init__.cpython-39.pyc
|   |   |   |-- augment.cpython-39.pyc
|   |   |   |-- base.cpython-39.pyc
|   |   |   |-- build.cpython-39.pyc
|   |   |   |-- converter.cpython-39.pyc
|   |   |   |-- dataset.cpython-39.pyc
|   |   |   |-- loaders.cpython-39.pyc
|   |   |   `-- utils.cpython-39.pyc
|   |   |-- annotator.py
|   |   |-- augment.py
|   |   |-- base.py
|   |   |-- build.py
|   |   |-- converter.py
|   |   |-- dataset.py
|   |   |-- explorer
|   |   |   |-- __init__.py
|   |   |   |-- __pycache__
|   |   |   |   |-- __init__.cpython-39.pyc
|   |   |   |   |-- explorer.cpython-39.pyc
|   |   |   |   `-- utils.cpython-39.pyc
|   |   |   |-- explorer.py
|   |   |   |-- gui
|   |   |   |   |-- __init__.py
|   |   |   |   `-- dash.py
|   |   |   `-- utils.py
|   |   |-- loaders.py
|   |   |-- scripts
|   |   |   |-- download_weights.sh
|   |   |   |-- get_coco.sh
|   |   |   |-- get_coco128.sh
|   |   |   `-- get_imagenet.sh
|   |   |-- split_dota.py
|   |   `-- utils.py
|   |-- engine
|   |   |-- __init__.py
|   |   |-- __pycache__
|   |   |   |-- __init__.cpython-39.pyc
|   |   |   |-- exporter.cpython-39.pyc
|   |   |   |-- model.cpython-39.pyc
|   |   |   |-- predictor.cpython-39.pyc
|   |   |   |-- results.cpython-39.pyc
|   |   |   |-- trainer.cpython-39.pyc
|   |   |   `-- validator.cpython-39.pyc
|   |   |-- exporter.py
|   |   |-- model.py
|   |   |-- predictor.py
|   |   |-- results.py
|   |   |-- trainer.py
|   |   |-- tuner.py
|   |   `-- validator.py
|   |-- hub
|   |   |-- __init__.py
|   |   |-- __pycache__
|   |   |   |-- __init__.cpython-39.pyc
|   |   |   |-- auth.cpython-39.pyc
|   |   |   `-- utils.cpython-39.pyc
|   |   |-- auth.py
|   |   |-- session.py
|   |   `-- utils.py
|   |-- models
|   |   |-- __init__.py
|   |   |-- __pycache__
|   |   |   `-- __init__.cpython-39.pyc
|   |   |-- fastsam
|   |   |   |-- __init__.py
|   |   |   |-- __pycache__
|   |   |   |   |-- __init__.cpython-39.pyc
|   |   |   |   |-- model.cpython-39.pyc
|   |   |   |   |-- predict.cpython-39.pyc
|   |   |   |   |-- prompt.cpython-39.pyc
|   |   |   |   |-- utils.cpython-39.pyc
|   |   |   |   `-- val.cpython-39.pyc
|   |   |   |-- model.py
|   |   |   |-- predict.py
|   |   |   |-- prompt.py
|   |   |   |-- utils.py
|   |   |   `-- val.py
|   |   |-- nas
|   |   |   |-- __init__.py
|   |   |   |-- __pycache__
|   |   |   |   |-- __init__.cpython-39.pyc
|   |   |   |   |-- model.cpython-39.pyc
|   |   |   |   |-- predict.cpython-39.pyc
|   |   |   |   `-- val.cpython-39.pyc
|   |   |   |-- model.py
|   |   |   |-- predict.py
|   |   |   `-- val.py
|   |   |-- rtdetr
|   |   |   |-- __init__.py
|   |   |   |-- __pycache__
|   |   |   |   |-- __init__.cpython-39.pyc
|   |   |   |   |-- model.cpython-39.pyc
|   |   |   |   |-- predict.cpython-39.pyc
|   |   |   |   |-- train.cpython-39.pyc
|   |   |   |   `-- val.cpython-39.pyc
|   |   |   |-- model.py
|   |   |   |-- predict.py
|   |   |   |-- train.py
|   |   |   `-- val.py
|   |   |-- sam
|   |   |   |-- __init__.py
|   |   |   |-- __pycache__
|   |   |   |   |-- __init__.cpython-39.pyc
|   |   |   |   |-- amg.cpython-39.pyc
|   |   |   |   |-- build.cpython-39.pyc
|   |   |   |   |-- model.cpython-39.pyc
|   |   |   |   `-- predict.cpython-39.pyc
|   |   |   |-- amg.py
|   |   |   |-- build.py
|   |   |   |-- model.py
|   |   |   |-- modules
|   |   |   |   |-- __init__.py
|   |   |   |   |-- __pycache__
|   |   |   |   |   |-- __init__.cpython-39.pyc
|   |   |   |   |   |-- decoders.cpython-39.pyc
|   |   |   |   |   |-- encoders.cpython-39.pyc
|   |   |   |   |   |-- sam.cpython-39.pyc
|   |   |   |   |   |-- tiny_encoder.cpython-39.pyc
|   |   |   |   |   `-- transformer.cpython-39.pyc
|   |   |   |   |-- decoders.py
|   |   |   |   |-- encoders.py
|   |   |   |   |-- sam.py
|   |   |   |   |-- tiny_encoder.py
|   |   |   |   `-- transformer.py
|   |   |   `-- predict.py
|   |   |-- utils
|   |   |   |-- __init__.py
|   |   |   |-- loss.py
|   |   |   `-- ops.py
|   |   |-- yolo
|   |   |   |-- __init__.py
|   |   |   |-- __pycache__
|   |   |   |   |-- __init__.cpython-39.pyc
|   |   |   |   `-- model.cpython-39.pyc
|   |   |   |-- classify
|   |   |   |   |-- __init__.py
|   |   |   |   |-- __pycache__
|   |   |   |   |   |-- __init__.cpython-39.pyc
|   |   |   |   |   |-- predict.cpython-39.pyc
|   |   |   |   |   |-- train.cpython-39.pyc
|   |   |   |   |   `-- val.cpython-39.pyc
|   |   |   |   |-- predict.py
|   |   |   |   |-- train.py
|   |   |   |   `-- val.py
|   |   |   |-- detect
|   |   |   |   |-- __init__.py
|   |   |   |   |-- __pycache__
|   |   |   |   |   |-- __init__.cpython-39.pyc
|   |   |   |   |   |-- predict.cpython-39.pyc
|   |   |   |   |   |-- train.cpython-39.pyc
|   |   |   |   |   `-- val.cpython-39.pyc
|   |   |   |   |-- predict.py
|   |   |   |   |-- train.py
|   |   |   |   `-- val.py
|   |   |   |-- model.py
|   |   |   |-- obb
|   |   |   |   |-- __init__.py
|   |   |   |   |-- __pycache__
|   |   |   |   |   |-- __init__.cpython-39.pyc
|   |   |   |   |   |-- predict.cpython-39.pyc
|   |   |   |   |   |-- train.cpython-39.pyc
|   |   |   |   |   `-- val.cpython-39.pyc
|   |   |   |   |-- predict.py
|   |   |   |   |-- train.py
|   |   |   |   `-- val.py
|   |   |   |-- pose
|   |   |   |   |-- __init__.py
|   |   |   |   |-- __pycache__
|   |   |   |   |   |-- __init__.cpython-39.pyc
|   |   |   |   |   |-- predict.cpython-39.pyc
|   |   |   |   |   |-- train.cpython-39.pyc
|   |   |   |   |   `-- val.cpython-39.pyc
|   |   |   |   |-- predict.py
|   |   |   |   |-- train.py
|   |   |   |   `-- val.py
|   |   |   `-- segment
|   |   |       |-- __init__.py
|   |   |       |-- __pycache__
|   |   |       |   |-- __init__.cpython-39.pyc
|   |   |       |   |-- predict.cpython-39.pyc
|   |   |       |   |-- train.cpython-39.pyc
|   |   |       |   `-- val.cpython-39.pyc
|   |   |       |-- predict.py
|   |   |       |-- train.py
|   |   |       `-- val.py
|   |   `-- yolov10
|   |       |-- __init__.py
|   |       |-- __pycache__
|   |       |   |-- __init__.cpython-39.pyc
|   |       |   |-- card.cpython-39.pyc
|   |       |   |-- model.cpython-39.pyc
|   |       |   |-- predict.cpython-39.pyc
|   |       |   |-- train.cpython-39.pyc
|   |       |   `-- val.cpython-39.pyc
|   |       |-- card.py
|   |       |-- model.py
|   |       |-- predict.py
|   |       |-- train.py
|   |       `-- val.py
|   |-- nn
|   |   |-- __init__.py
|   |   |-- __pycache__
|   |   |   |-- __init__.cpython-39.pyc
|   |   |   |-- autobackend.cpython-39.pyc
|   |   |   `-- tasks.cpython-39.pyc
|   |   |-- autobackend.py
|   |   |-- modules
|   |   |   |-- __init__.py
|   |   |   |-- __pycache__
|   |   |   |   |-- __init__.cpython-39.pyc
|   |   |   |   |-- block.cpython-39.pyc
|   |   |   |   |-- conv.cpython-39.pyc
|   |   |   |   |-- head.cpython-39.pyc
|   |   |   |   |-- transformer.cpython-39.pyc
|   |   |   |   `-- utils.cpython-39.pyc
|   |   |   |-- block.py
|   |   |   |-- conv.py
|   |   |   |-- head.py
|   |   |   |-- transformer.py
|   |   |   `-- utils.py
|   |   `-- tasks.py
|   |-- solutions
|   |   |-- __init__.py
|   |   |-- ai_gym.py
|   |   |-- distance_calculation.py
|   |   |-- heatmap.py
|   |   |-- object_counter.py
|   |   `-- speed_estimation.py
|   |-- trackers
|   |   |-- README.md
|   |   |-- __init__.py
|   |   |-- basetrack.py
|   |   |-- bot_sort.py
|   |   |-- byte_tracker.py
|   |   |-- track.py
|   |   `-- utils
|   |       |-- __init__.py
|   |       |-- gmc.py
|   |       |-- kalman_filter.py
|   |       `-- matching.py
|   `-- utils
|       |-- __init__.py
|       |-- __pycache__
|       |   |-- __init__.cpython-39.pyc
|       |   |-- autobatch.cpython-39.pyc
|       |   |-- checks.cpython-39.pyc
|       |   |-- dist.cpython-39.pyc
|       |   |-- downloads.cpython-39.pyc
|       |   |-- files.cpython-39.pyc
|       |   |-- instance.cpython-39.pyc
|       |   |-- loss.cpython-39.pyc
|       |   |-- metrics.cpython-39.pyc
|       |   |-- ops.cpython-39.pyc
|       |   |-- patches.cpython-39.pyc
|       |   |-- plotting.cpython-39.pyc
|       |   |-- tal.cpython-39.pyc
|       |   `-- torch_utils.cpython-39.pyc
|       |-- autobatch.py
|       |-- benchmarks.py
|       |-- callbacks
|       |   |-- __init__.py
|       |   |-- __pycache__
|       |   |   |-- __init__.cpython-39.pyc
|       |   |   |-- base.cpython-39.pyc
|       |   |   `-- hub.cpython-39.pyc
|       |   |-- base.py
|       |   |-- clearml.py
|       |   |-- comet.py
|       |   |-- dvc.py
|       |   |-- hub.py
|       |   |-- mlflow.py
|       |   |-- neptune.py
|       |   |-- raytune.py
|       |   |-- tensorboard.py
|       |   `-- wb.py
|       |-- checks.py
|       |-- dist.py
|       |-- downloads.py
|       |-- errors.py
|       |-- files.py
|       |-- instance.py
|       |-- loss.py
|       |-- metrics.py
|       |-- ops.py
|       |-- patches.py
|       |-- plotting.py
|       |-- tal.py
|       |-- torch_utils.py
|       |-- triton.py
|       `-- tuner.py
`-- ultralytics.egg-info
    |-- PKG-INFO
    |-- SOURCES.txt
    |-- dependency_links.txt
    |-- entry_points.txt
    |-- requires.txt
    `-- top_level.txt
```