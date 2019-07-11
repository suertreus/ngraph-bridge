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

#ifndef NGRAPH_LOG_H_
#define NGRAPH_LOG_H_

#include <string>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

namespace ngraph_bridge {

int64 MinNGraphVLogLevel();

}  // namespace ngraph_bridge

}  // namespace tensorflow

#define NGRAPH_VLOG_IS_ON(lvl) ((lvl) <= ::tensorflow::ngraph_bridge::MinNGraphVLogLevel())

#define NGRAPH_VLOG(lvl)                  \
  switch (NGRAPH_VLOG_IS_ON(lvl) ? 1 : 0) \
  case 1:                                 \
    LOG(INFO)

#endif  // NGRAPH_LOG_H_
