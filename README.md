
# Depth Perception using Stereo Imaging

## Project Overview

This project implements the concept of Stereo Vision, which is a technique used to infer depth information from two-dimensional images. By utilizing two images of the same scene captured from different camera angles (simulating human binocular vision), we calculate the relative positions of objects to extract 3D depth information.

The project works with three datasets—each containing image pairs from two vantage points. By comparing the pixel differences between these paired images, we estimate the distance to objects in the scene.

## Datasets

There are three datasets used in this project:
- **curule**
- **octagon**
- **pendulum**

Each dataset contains a pair of images taken from two different perspectives of the same scene. By applying stereo vision techniques to these images, we calculate the disparity between them and generate depth maps.

## Project Structure

```bash
├── datasets/
│   ├── curule/
│   ├── octagon/
│   ├── pendulum/
├── solution.py       # Main script to compute depth maps
├── utils.py          # Utility functions for stereo vision computation
├── README.md         # Detailed documentation (this file)
└── requirements.txt  # Python dependencies
```

## Authors

- [@usnikchawla](https://github.com/usnikchawla)

## Dependencies

To run the project, you need the following Python libraries:

- `numpy`
- `opencv-python`
- `matplotlib`

You can install them using `pip`:

```bash
pip install -r requirements.txt
```

## How It Works

The project performs the following steps to compute the depth information:

1. **Preprocess the images**: The two images are loaded, resized, and converted to grayscale for further processing.
2. **Feature matching**: Key features from both images are matched using algorithms such as SIFT, SURF, or ORB.
3. **Disparity calculation**: The differences (disparity) between corresponding pixels in the two images are computed.
4. **Depth map generation**: Based on the disparity values, a depth map is created, representing the distance of each point in the image from the camera.

The `utils.py` file contains helper functions to handle image preprocessing and disparity calculation. 

## Running the Project

To deploy this project, run the following command from the terminal:

```bash
python3 path/to/solution.py folder_in_data
```

Here, `folder_in_data` represents the dataset you wish to use and can have one of the following values:
- `curule`
- `octagon`
- `pendulum`

For example:

```bash
python3 solution.py curule
```

### Notes:
- Ensure that `utils.py` is in the same directory as `solution.py`.
- The datasets should be structured as shown in the **Project Structure** section above.

## Example Output

The output will consist of a disparity map and a depth map, which represent the 3D information extracted from the image pair. The depth map visually encodes the distance to objects in the scene, with darker regions representing closer objects and lighter regions representing further objects.

### Sample Disparity Map:
![disparity_map](https://example.com/sample_disparity.png)

### Sample Depth Map:
![depth_map](https://example.com/sample_depth.png)

## Future Improvements

- Implementing more advanced stereo matching algorithms for improved accuracy.
- Adding options for different types of stereo rectification methods.
- Handling dynamic scenes with moving objects.
  
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
