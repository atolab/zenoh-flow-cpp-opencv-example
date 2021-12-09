//
// Copyright (c) 2017, 2021 ADLINK Technology Inc.
//
// This program and the accompanying materials are made available under the
// terms of the Eclipse Public License 2.0 which is available at
// http://www.eclipse.org/legal/epl-2.0, or the Apache License, Version 2.0
// which is available at https://www.apache.org/licenses/LICENSE-2.0.
//
// SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
//
// Contributors:
//   ADLINK zenoh team, <zenoh@adlink-labs.tech>
//

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <operator.hpp>

#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>


#include <nlohmann/json.hpp>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

// for convenience
using json = nlohmann::json;

namespace zenoh {
namespace flow {

const cv::Scalar colors[] = {
    {0, 255, 0},
    {255, 255, 0},
    {0, 255, 255},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors)/sizeof(colors[0]);
constexpr float CONFIDENCE_THRESHOLD = 0;
constexpr float NMS_THRESHOLD = 0.4;


State::State(std::string net_cfg, std::string net_weights, std::string net_classes) {
  this->classes = std::vector<std::string>{};
  this->dnn = cv::dnn::readNetFromDarknet(net_cfg, net_weights);
  this->dnn.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  this->dnn.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

  std::ifstream class_file(net_classes);
  if (!class_file) {
    std::cerr << "failed to open %s" << net_classes << std::endl;
  }
  std::string line;
  while (std::getline(class_file, line)) this->classes.push_back(line);

  this->outputs = this->dnn.getUnconnectedOutLayersNames();

  this->num_classes = this->classes.size();
  this->flags = { 0 };
}

std::vector<int> State::getFlags(void) {
	return this->flags;
}

std::vector<std::string> State::getClasses(void) {
  return this->classes;
}

cv::dnn::Net State::getDNN(void) {
  return this->dnn;
}

std::vector<std::string> State::getOutputs(void) {
  return this->outputs;
}

std::size_t State::getNumClasses(void) {
  return this->num_classes;
}

std::unique_ptr<State> initialize(rust::Str json_configuration) {
  std::string net_cfg;
  std::string net_weights;
  std::string net_classes;

  json config;

  config = json::parse(json_configuration);

  if (config.contains("neural-network")) {
    net_cfg = config["neural-network"].get<std::string>();
  }

  if (config.contains("network-weights")) {
    net_weights = config["network-weights"].get<std::string>();
  }

  if (config.contains("network-classes")) {
    net_classes = config["network-classes"].get<std::string>();
  }

  return std::make_unique<State>(net_cfg, net_weights, net_classes);

}

std::vector<std::uint8_t> getFrameInput(rust::Vec<Input> inputs) {
  std::vector<std::uint8_t> buffer;
  for (auto input: inputs) {
    if (input.port_id == "Frame") {
      for (auto byte : input.data) {
        buffer.push_back(byte);
      }
    }
  }
  return buffer;
}

rust::Vec<std::uint8_t> makeFrameOutput(std::vector<std::uint8_t> frame) {

  rust::Vec<std::uint8_t> buffer{};
  for (auto byte : frame) {
    buffer.push_back(byte);
  }
  return buffer;
}

bool
input_rule(Context &context, std::unique_ptr<State> &state, rust::Vec<Token> &tokens) {
  (void)context;
  (void)state;

  for (auto token : tokens) {
    if (token.status != TokenStatus::Ready) {
        return false;
      }
  }

  return true;
}

rust::Vec<Output>
run(Context &context, std::unique_ptr<State> &state, rust::Vec<Input> inputs) {
  (void)context;

  auto scale = 1.0/255.0;
  cv::Scalar mean = {0,0,0,0};

  auto net = state->getDNN();

  cv::Mat frame, blob;
  std::vector<cv::Mat> detections;
  std::vector<std::uint8_t> result;

  auto buffer = getFrameInput(inputs);

  cv::imdecode(buffer, cv::IMREAD_COLOR, &frame);

  cv::dnn::blobFromImage(frame, blob, scale, cv::Size(512,512), mean,  true, false, CV_32F);
  net.setInput(blob, "", 1.0, mean);

  auto start = std::chrono::high_resolution_clock::now();
  net.forward(detections, state->getOutputs());
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long elapsed_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();



  std::vector<int> indices[state->getNumClasses()];
  std::vector<cv::Rect> boxes[state->getNumClasses()];
  std::vector<float> scores[state->getNumClasses()];

  for (auto& output : detections) {
    const auto num_boxes = output.rows;
    for (int i = 0; i < num_boxes; i++) {
      auto x = output.at<float>(i, 0) * frame.cols;
      auto y = output.at<float>(i, 1) * frame.rows;
      auto width = output.at<float>(i, 2) * frame.cols;
      auto height = output.at<float>(i, 3) * frame.rows;
      cv::Rect rect(x - width/2, y - height/2, width, height);

      for (size_t c = 0; c < state->getNumClasses(); c++) {
          auto confidence = *output.ptr<float>(i, 5 + c);
          if (confidence >= CONFIDENCE_THRESHOLD) {
            boxes[c].push_back(rect);
            scores[c].push_back(confidence);
          }
      }
    }
  }

  for (size_t c = 0; c < state->getNumClasses(); c++) cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

  int detected = 0;

  for (size_t c= 0; c < state->getNumClasses(); c++) {
    for (size_t i = 0; i < indices[c].size(); ++i) {
      const auto color = colors[c % NUM_COLORS];
      auto idx = indices[c][i];
      const auto& rect = boxes[c][idx];
      cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

      std::ostringstream label_ss;
      label_ss << state->getClasses()[c] << ": " << scores[c][idx];
      auto label = label_ss.str();

      int baseline;
      auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
      cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
      cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
      detected += 1;
    }
  }

  std::ostringstream label_stats;
  label_stats << "DNN Inference time: " << elapsed_microseconds << "us - Detected: " << detected;
  auto label_s = label_stats.str();

  auto bg_size = cv::getTextSize(label_s, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, 1, 0);

  cv::rectangle(frame, cv::Point(0, 0), cv::Point(bg_size.width, bg_size.height+10), cv::Scalar(0,0,0), cv::FILLED);
  cv::putText(frame, label_s.c_str(), cv::Point(0, bg_size.height+5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 0));

  cv::imencode(".jpg", frame, result);

  rust::Vec<std::uint8_t> enc_frame = makeFrameOutput(result);
  Output res { "Frame", enc_frame};
  rust::Vec<Output> results { res };
  return results;
}

  rust::Vec<Output>
  output_rule(Context &context, std::unique_ptr<State> &state, rust::Vec<Output> run_outputs, LocalDeadlineMiss deadlinemiss) {
    return run_outputs;
  }
} // namespace flow
} // namespace zenoh
