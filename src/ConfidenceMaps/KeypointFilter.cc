#include <iostream>
#include <filesystem> 
#include "KeypointFilter.h"

torch::jit::script::Module globalModel;
static std::once_flag modelLoadFlag;

// Function to load the model
void loadModel() {
    try {
        globalModel = torch::jit::load("src/ConfidenceMaps/confidences_model.pt");
        globalModel.to(torch::kCUDA);
        torch::NoGradGuard no_grad;
        std::cout << "Model loaded successfully." << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        throw std::runtime_error("Failed to load the model");
    }
}

torch::Tensor convertMatToTensor(const cv::Mat& image) {
    
    // Convert cv::Mat to a float tensor    
    cv::Mat image_float;
    image.convertTo(image_float, CV_32F, 1.0 / 255.0);  // Normalize the image to [0, 1]
    
    // Change data layout from HxWxC to CxHxW required by PyTorch
    torch::Tensor tensor_image = torch::from_blob(image_float.data, 
        {image.rows, image.cols, 1}, torch::kFloat32).clone(); // Copy the data
    tensor_image = tensor_image.permute({2, 0, 1}); // HxWxC to CxHxW
    
    // Add batch dimension
    return tensor_image.unsqueeze(0);
}

// Implementation of the FilterKeyPoints function
void FilterKeyPoints(std::vector<cv::KeyPoint>& _keypoints, cv::Mat& descriptors, cv::Mat image, const std::string& confidenceLevel) {
    
    std::call_once(modelLoadFlag, loadModel);
    std::cout << "Using the model in recurrent function..." << std::endl;

    torch::Tensor image_tensor = convertMatToTensor(image);
    image_tensor = image_tensor.to(torch::kCUDA);
    // Run the model
    auto outputs = globalModel.forward({image_tensor}).toTuple();

    auto features = outputs->elements()[0].toList();
    auto confs = outputs->elements()[1].toList();

    // Determine the size for interpolation
    auto shape = confs.get(0).toTensor().sizes();
    std::vector<torch::Tensor> resized_confs;
    for (size_t i = 0; i < confs.size(); ++i) {
        torch::Tensor c = confs.get(i).toTensor();  // Corrected: Use toIValue() before toTensor()

        auto options = torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>({shape[1], shape[2]})) // use spatial dimensions
            .mode(torch::kBilinear)
            .align_corners(false);

        auto resized = torch::nn::functional::interpolate(c.unsqueeze(0), options);
        resized_confs.push_back(resized.squeeze(0)); // Remove batch dimension after resizing
    }

    // Extract fine, mid, coarse tensors
    torch::Tensor fine = resized_confs[0][0];
    torch::Tensor mid = resized_confs[1][0];
    torch::Tensor coarse = resized_confs[2][0];

    // Fuse the confidence maps (for visualization only)
    torch::Tensor fused = torch::pow(fine * mid * coarse, 1.0 / 3.0);

    // If needed, convert the tensor back to cv::Mat to display with OpenCV
    // This part is omitted, as displaying or handling the tensor data in OpenCV would require more context

    std::cout << "Initial keypoints count: " << _keypoints.size() << std::endl;
    
    torch::Tensor level;
    if (confidenceLevel == "fused"){
        level = fused;
    }
    else if (confidenceLevel == "coarse"){
        level = coarse;
    }
    else if (confidenceLevel == "medium"){
        level = mid;
    }
    else{
        level = fine;
    }

    torch::Tensor median_tensor = level.median();
    float threshold = median_tensor.item<float>();
    // Round to two decimal places
    float rounded_median = std::round(threshold * 100.0) / 100.0;

    // Printing the results
    std::cout << "Median value: " << threshold << std::endl;
    std::cout << "Rounded Median: " << rounded_median << std::endl;


    level = level.to(torch::kCPU).contiguous();

    auto confidence = level.accessor<float, 2>(); 
            
    std::vector<cv::KeyPoint> filteredKeypoints;
    cv::Mat filteredDescriptors;

    for (size_t i = 0; i < _keypoints.size(); ++i)  {
        cv::KeyPoint kp = _keypoints[i];
        int x = static_cast<int>(std::round(kp.pt.x));
        int y = static_cast<int>(std::round(kp.pt.y));

        // Check bounds
        if (y >= 0 && y < confidence.size(0) && x >= 0 && x < confidence.size(1)) {
            float confValue = confidence[y][x];
            if (confValue > threshold) {
                filteredKeypoints.push_back(kp);
                filteredDescriptors.push_back(descriptors.row(i));
            }
        }
    }

    // Replace the original keypoints with the filtered ones
    _keypoints.swap(filteredKeypoints);
    descriptors = filteredDescriptors.clone();
    std::cout << "Filtered keypoints count: " << _keypoints.size() << std::endl;
    std::cout << "Processing completed.\n";
}