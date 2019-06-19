/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "ngraph_backend_manager.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

BackendManager::~BackendManager() {
  NGRAPH_VLOG(2) << "BackendManager::~BackendManager()";
}

// initialize backend manager
string BackendManager::ng_backend_name_ = "CPU";
mutex BackendManager::ng_backend_name_mutex_;
map<string, Backend*> BackendManager::ng_backend_map_;
mutex BackendManager::ng_backend_map_mutex_;
map<std::string, int> BackendManager::ref_count_each_backend_;

mutex BackendManager::ng_backendconfig_map_mutex_;
unordered_map<string, std::unique_ptr<BackendConfig>>
    BackendManager::ng_backendconfig_map_;

Status BackendManager::SetBackendName(const string& backend_name) {
  std::lock_guard<std::mutex> lock(BackendManager::ng_backend_name_mutex_);
  if (backend_name.empty() || !IsSupportedBackend(backend_name)) {
    return errors::Internal("Backend ", backend_name,
                            " is not supported on nGraph");
  }
  BackendManager::ng_backend_name_ = backend_name;
  return Status::OK();
}

Status BackendManager::CreateBackend(const string& backend_name) {
  std::lock_guard<std::mutex> lock(BackendManager::ng_backend_map_mutex_);
  auto itr = BackendManager::ng_backend_map_.find(backend_name);
  // if backend does not exist create it
  if (itr == BackendManager::ng_backend_map_.end()) {
    Backend* bend = new Backend;
    std::shared_ptr<ng::runtime::Backend> bend_ptr;
    try {
      bend_ptr = ng::runtime::Backend::create(backend_name);
    } catch (const std::exception& e) {
      return errors::Internal("Could not create backend of type ", backend_name,
                              ". Got exception ", e.what());
    }

    if (bend_ptr == nullptr) {
      return errors::Internal("Could not create backend of type ",
                              backend_name);
    }
    bend->backend_ptr = std::move(bend_ptr);
    BackendManager::ng_backend_map_[backend_name] = bend;
    BackendManager::ref_count_each_backend_[backend_name] = 0;
  }
  BackendManager::ref_count_each_backend_[backend_name]++;

  NGRAPH_VLOG(2) << "BackendManager::CreateBackend(): " << backend_name
                 << " ref_count: "
                 << BackendManager::ref_count_each_backend_[backend_name];
  return Status::OK();
}

void BackendManager::ReleaseBackend(const string& backend_name) {
  std::lock_guard<std::mutex> lock(BackendManager::ng_backend_map_mutex_);
  BackendManager::ref_count_each_backend_[backend_name]--;
  NGRAPH_VLOG(2) << "BackendManager::ReleaseBackend(): " << backend_name
                 << " ref_count: "
                 << BackendManager::ref_count_each_backend_[backend_name];
  if (BackendManager::ref_count_each_backend_[backend_name] == 0) {
    BackendManager::ng_backend_map_[backend_name]->backend_ptr.reset();
    BackendManager::ng_backend_map_.erase(backend_name);
    NGRAPH_VLOG(2) << "Deleted Backend " << backend_name;
  }
}

// Returns a backend pointer of the type specified by the backend name
ng::runtime::Backend* BackendManager::GetBackend(const string& backend_name) {
  return BackendManager::ng_backend_map_.at(backend_name)->backend_ptr.get();
}

// LockBackend
void BackendManager::LockBackend(const string& backend_name) {
  BackendManager::ng_backend_map_.at(backend_name)->backend_mutex.lock();
}

// UnlockBackend
void BackendManager::UnlockBackend(const string& backend_name) {
  BackendManager::ng_backend_map_.at(backend_name)->backend_mutex.unlock();
}

// Returns the nGraph supported backend names
vector<string> BackendManager::GetSupportedBackendNames() {
  return ng::runtime::BackendManager::get_registered_backends();
}

size_t BackendManager::GetNumOfSupportedBackends() {
  return ng::runtime::BackendManager::get_registered_backends().size();
}

bool BackendManager::IsSupportedBackend(const string& backend_name) {
  auto status = BackendManager::CreateBackend(backend_name);
  if (status != Status::OK()) {
    return false;
  }
  return true;
};

Status BackendManager::GetCurrentlySetBackendName(string* backend_name) {
  const char* ng_backend_env_value = std::getenv("NGRAPH_TF_BACKEND");

  // NGRAPH_TF_BACKEND is not set
  if (ng_backend_env_value == nullptr) {
    *backend_name = BackendManager::ng_backend_name_;
    NGRAPH_VLOG(1) << "Using the currently set backend " << (*backend_name);
    return Status::OK();
  }

  // NGRAPH_TF_BACKEND is set
  string backend_env = std::string(ng_backend_env_value);
  if (backend_env.empty() || !BackendManager::IsSupportedBackend(backend_env)) {
    return errors::Internal("NGRAPH_TF_BACKEND: ", backend_env,
                            " is not supported");
  }

  *backend_name = backend_env;
  NGRAPH_VLOG(1) << "Overriding backend using the enviornment variable "
                    "to "
                 << (*backend_name);
  return Status::OK();
};

// Backend Config functions
// BackendConfig is expected to be a readonly class
// hence only locked at creation and not during later access
std::unique_ptr<BackendConfig>& BackendManager::GetBackendConfig(
    const string& backend_name) {
  std::lock_guard<std::mutex> lock(BackendManager::ng_backend_map_mutex_);
  auto itr = BackendManager::ng_backendconfig_map_.find(backend_name);
  if (itr == BackendManager::ng_backendconfig_map_.end()) {
    if (backend_name == "NNPI") {
      BackendManager::ng_backendconfig_map_.insert(std::make_pair(
          backend_name,
          std::unique_ptr<BackendNNPIConfig>(new BackendNNPIConfig())));
    } else {
      BackendManager::ng_backendconfig_map_.insert(std::make_pair(
          backend_name,
          std::unique_ptr<BackendConfig>(new BackendConfig(backend_name))));
    }
  }
  return BackendManager::ng_backendconfig_map_.at(backend_name);
}

vector<string> BackendManager::GetBackendAdditionalAttributes(
    const string& backend_name) {
  return BackendManager::GetBackendConfig(backend_name)
      ->GetAdditionalAttributes();
}

unordered_map<string, string> BackendManager::GetBackendAttributeValues(
    const string& backend_config) {
  unordered_map<string, string> backend_parameters;

  string backend_name = backend_config.substr(0, backend_config.find(':'));
  NGRAPH_VLOG(3) << "Got Backend Name " << backend_name;

  return BackendManager::GetBackendConfig(backend_name)->Split(backend_config);
}

string BackendManager::GetBackendCreationString(
    const string& backend_name,
    const unordered_map<string, string>& additional_attribute_map) {
  return BackendManager::GetBackendConfig(backend_name)
      ->Join(additional_attribute_map);
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
