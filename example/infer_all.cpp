/* Copyright (c) 2019 Xieyuanli Chen, Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of rangenet_lib, and covered by the provided LICENSE file.
 *
 */


// opencv stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>

// c++ stuff
#include <chrono>
#include <iomanip>  // for setfill
#include <iostream>
#include <string>
#include <vector>

// net stuff
#include <selector.hpp>
namespace cl = rangenet::segmentation;

// // standalone lib h
// #include "infer.hpp"

// boost
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

typedef std::tuple< u_char, u_char, u_char> color;

int main(int argc, const char *argv[]) {
  // define options
  std::string model_path;
  std::string scan_dir;
  std::string backend = "tensorrt";
  bool verbose = false;

  // Parse options
  try {
    po::options_description desc{"Options"};
    desc.add_options()("help,h", "Help screen")(
        "model_path,m", po::value<std::string>(&model_path),
        "Directory to get the inference model from. No default")(
        "scan_dir,s", po::value<std::string>(&scan_dir),
        "Directory containing LiDAR scans to infer. No default")(
        "verbose,v", po::bool_switch(),
        "Verbose mode. Calculates profile (time to run)");

    po::variables_map vm;
    po::store(parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }

    // check if model path and scan directory are provided
    if (vm.count("model_path")) {
      model_path = vm["model_path"].as<std::string>() + "/";  // make sure path is valid
      std::cout << "model_path: " << model_path << std::endl;
    } else {
      std::cerr << "No model path! See --help (-h) for help. Exiting" << std::endl;
      return 1;
    }

    if (vm.count("scan_dir")) {
      scan_dir = vm["scan_dir"].as<std::string>() + "/";  // make sure path is valid
      std::cout << "scan_dir: " << scan_dir << std::endl;
    } else {
      std::cerr << "No scan directory! See --help (-h) for help. Exiting" << std::endl;
      return 1;
    }

    if (vm.count("verbose")) {
      verbose = vm["verbose"].as<bool>();
      std::cout << "verbose: " << verbose << std::endl;
    } else {
      std::cout << "verbose: " << verbose << ". Using default!" << std::endl;
    }

    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
  } catch (const po::error &ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }

  // create a network
  std::unique_ptr<cl::Net> net = cl::make_net(model_path, backend);

  // set verbosity
  net->verbosity(verbose);

  // iterate over all scan files in the directory
  fs::path scan_path(scan_dir);
  if (!fs::is_directory(scan_path)) {
    std::cerr << "Invalid scan directory! See --help (-h) for help. Exiting" << std::endl;
    return 1;
  }

  // std::vector<std::string> scan_files;
  std::map<int, std::string> scan_files; // 存储文件名数字部分与文件路径的映射
  for (auto& entry : fs::directory_iterator(scan_path)) {
    if (fs::is_regular_file(entry) && entry.path().extension() == ".bin") {
      // scan_files.push_back(entry.path().string());
      std::string filename = entry.path().stem().string();
      int file_number = std::stoi(filename); // 将文件名转换为数字
      scan_files[file_number] = entry.path().string(); // 将文件名数字部分与文件路径关联起来
    }
  }

  if (scan_files.empty()) {
    std::cerr << "No scan files found in the directory! Exiting" << std::endl;
    return 1;
  }

  std::cout << "Found " << scan_files.size() << " scan files in directory." << std::endl;
  std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
  cv::viz::Viz3d window("semantic scan");

  for (const auto& scan : scan_files) {
    // predict each image
    std::cout << "Predicting image: " << scan.second << std::endl;

    // Open a scan
    std::ifstream in(scan.second.c_str(), std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Could not open the scan: " << scan.second << std::endl;
        continue;
    }

    in.seekg(0, std::ios::end);
    uint32_t num_points = in.tellg() / (4 * sizeof(float));
    in.seekg(0, std::ios::beg);

    std::vector<float> values(4 * num_points);
    in.read((char*)&values[0], 4 * num_points * sizeof(float));

    // predict
    std::vector<std::vector<float>> semantic_scan = net->infer(values, num_points);

    // get point cloud
    std::vector<cv::Vec3f> points = net->getPoints(values, num_points);

    // get color mask
    std::vector<cv::Vec3b> color_mask = net->getLabels(semantic_scan, num_points);

    // print the output
    if (verbose) {
      // cv::viz::Viz3d window("semantic scan");
      cv::viz::WCloud cloudWidget(points, color_mask);
      while (!window.wasStopped()) {
        window.showWidget("cloud", cloudWidget);
        window.spinOnce(100, true);
        break;
      }
    }
    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
  }

  std::cout << "Example finished! "<< std::endl;

  return 0;
}
