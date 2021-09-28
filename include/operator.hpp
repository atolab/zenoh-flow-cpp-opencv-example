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

#pragma once
#include <vector>
#include <string>
#include <opencv2/dnn/dnn.hpp>
#include <wrapper.hpp>


namespace zenoh {
namespace flow {

class State {
private:
	std::vector<std::string> classes;
	cv::dnn::Net dnn;
	std::vector<int> flags;
	std::vector<std::string> outputs;
  size_t num_classes;

public:
  State(std::string net_cfg, std::string net_weights, std::string net_classes);
  std::vector<int> getFlags(void);
  std::vector<std::string> getClasses(void);
  cv::dnn::Net getDNN(void);
  std::vector<std::string> getOutputs(void);
  std::size_t getNumClasses(void);

};

std::unique_ptr<State> initialize(const ConfigurationMap &configuration);
bool input_rule(Context &context, std::unique_ptr<State> &state,
                rust::Vec<Token> &tokens);
rust::Vec<Output> run(Context &context,
                      std::unique_ptr<State> &state,
                      rust::Vec<Input> inputs);

} // namespace flow
} // namespace zenoh
