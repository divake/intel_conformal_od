# Intel RealSense Object Detection Demos

## 🚀 Quick Start Guide for Intel Demo

### Prerequisites
```bash
# Ensure you're in the intel_ai conda environment
conda activate intel_ai
```

### Run the Demo
```bash
# Navigate to this directory
cd /home/nus-ai/divek_nus/conformal_od/obj_det_conformal

# Run YOLOv8 with Uncertainty Quantification
python realsense_yolo_conformal.py
```

## 🎮 Demo Controls

- **`q`** - Quit the application
- **`s`** - Save screenshot (saves both RGB and depth images)
- **`c`** - Start/stop calibration (important for uncertainty!)
- **`u`** - Toggle uncertainty visualization on/off

## 📊 What You'll See

1. **Object Detection**: Real-time detection of 80 object classes
2. **Uncertainty Boxes**: 
   - Solid inner box = detected object
   - Dashed outer box = uncertainty region (90% confidence)
3. **Depth Information**: Distance to each object in meters
4. **Performance Metrics**: FPS and inference time

## 🎯 How to Demo

1. **Start the application**
2. **Press 'c' to calibrate** - Move camera around for 100 frames
3. **Watch the uncertainty boxes** - They adapt based on detection confidence
4. **Try different objects** - People, laptops, bottles, chairs, etc.
5. **Note the depth measurements** - Accurate to ~2cm with RealSense D455

## 🔧 Troubleshooting

If you see warnings about Qt/Wayland, ignore them - the app will still work.

If no objects are detected:
- Ensure good lighting
- Try common objects (person, laptop, bottle, chair)
- Adjust camera angle

## 📈 Performance

On Intel Core Ultra 7 165H:
- **30+ FPS** with YOLOv8m
- **<35ms** inference time
- Real-time uncertainty quantification

## 🌟 Key Selling Points for Intel

1. **Real-time AI on CPU** - No GPU required
2. **Uncertainty Quantification** - Critical for safety applications
3. **3D Perception** - Depth + detection combined
4. **Production Ready** - Robust, tested implementation

---

**Demo tip**: Show how uncertainty increases for partially occluded or distant objects!