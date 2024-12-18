// *************************************************************************
/* 
 * This file is part of the slamplay project.
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version, at your option. If this file is a modified/adapted 
 * version of an original file distributed under a different license that 
 * is not compatible with the GNU General Public License, the 
 * BSD 3-Clause License will apply instead.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
// *************************************************************************
#include "features_dl/hfnet/HFNetBaseModel.h"
#include "features_dl/hfnet/HFNetRTModel.h"
#include "features_dl/hfnet/HFNetSettings.h"
#include "features_dl/hfnet/HFNetTFModel.h"
#include "features_dl/hfnet/HFNetTFModelV2.h"
#include "features_dl/hfnet/HFNetVINOModel.h"

#include <unordered_map>
#include <unordered_set>

using namespace std;

namespace hfnet {

std::vector<HFNetBaseModel*> gvpModels;
HFNetBaseModel* gpGlobalModel = nullptr;

void InitAllModels(HFNetSettings* settings) {
    InitAllModels(settings->strModelPath(), settings->modelType(), settings->imageSize(), settings->nLevels(), settings->scaleFactor());
}

void InitAllModels(const std::string& strModelPath, ModelType modelType, cv::Size ImSize, int nLevels, float scaleFactor) {
    // Init Local Models
    if (gvpModels.size())
    {
        for (auto pModel : gvpModels) delete pModel;
        gvpModels.clear();
    }

    gvpModels.reserve(nLevels);
    float scale = 1.0f;
    for (int level = 0; level < nLevels; ++level)
    {
        cv::Vec4i inputShape{1, cvRound(ImSize.height * scale), cvRound(ImSize.width * scale), 1};
        HFNetBaseModel* pNewModel = nullptr;
        ModelDetectionMode mode;
        if (modelType == kHFNetTFModel)
        {
            if (level == 0)
                mode = kImageToLocalAndIntermediate;
            else
                mode = kImageToLocal;
            pNewModel = InitTFModel(strModelPath, mode, inputShape);
        } else if (modelType == kHFNetRTModel)
        {
            if (level == 0)
                mode = kImageToLocalAndGlobal;
            else
                mode = kImageToLocal;
            pNewModel = InitRTModel(strModelPath, mode, inputShape);
        } else if (modelType == kHFNetVINOModel)
        {
            if (level == 0)
                mode = kImageToLocalAndIntermediate;
            else
                mode = kImageToLocal;
            pNewModel = InitVINOModel(strModelPath + "/local_part", mode, inputShape);
        } else
        {
            cerr << "Wrong type of model!" << endl;
            exit(-1);
        }
        gvpModels.emplace_back(pNewModel);
        scale /= scaleFactor;
    }

    // Init Global Model
    if (gpGlobalModel) delete gpGlobalModel;

    cv::Vec4i inputShape{1, ImSize.height / 8, ImSize.width / 8, 96};
    HFNetBaseModel* pNewModel = nullptr;
    ModelDetectionMode mode;
    if (modelType == kHFNetTFModel)
    {
        mode = kIntermediateToGlobal;
        pNewModel = InitTFModel(strModelPath, mode, inputShape);
    } else if (modelType == kHFNetRTModel)
    {
        pNewModel = nullptr;
    } else if (modelType == kHFNetVINOModel)
    {
        mode = kIntermediateToGlobal;
        pNewModel = InitVINOModel(strModelPath + "/global_part", mode, inputShape);
    } else
    {
        cerr << "Wrong type of model!" << endl;
        exit(-1);
    }
    gpGlobalModel = pNewModel;
}

std::vector<HFNetBaseModel*> GetModelVec(void) {
    if (gvpModels.empty())
    {
        cerr << "Try to get models before initialize them" << endl;
        exit(-1);
    }
    return gvpModels;
}

HFNetBaseModel* GetGlobalModel(void) {
    if (gvpModels.empty())
    {
        cerr << "Try to get global model before initialize it" << endl;
        exit(-1);
    }
    return gpGlobalModel;
}

HFNetBaseModel* InitTFModel(const std::string& strModelPath, ModelDetectionMode mode, cv::Vec4i inputShape) {
    HFNetBaseModel* pModel;
    pModel = new HFNetTFModelV2(strModelPath, mode, inputShape);
    if (pModel->IsValid())
    {
        cout << "Successfully loaded HFNet TensorFlow model."
             << " Mode: " << gStrModelDetectionName[mode]
             << " Shape: " << inputShape.t() << endl;
    } else
        exit(-1);

    return pModel;
}

HFNetBaseModel* InitRTModel(const std::string& strModelPath, ModelDetectionMode mode, cv::Vec4i inputShape) {
    HFNetBaseModel* pModel;
    pModel = new HFNetRTModel(strModelPath, mode, inputShape);
    if (pModel->IsValid())
    {
        cout << "Successfully loaded HFNet TensorRT model."
             << " Mode: " << gStrModelDetectionName[mode]
             << " Shape: " << inputShape.t() << endl;
    } else
        exit(-1);

    return pModel;
}

HFNetBaseModel* InitVINOModel(const std::string& strModelPath, ModelDetectionMode mode, cv::Vec4i inputShape) {
    HFNetBaseModel* pModel;
    string strXmlPath = strModelPath + "/saved_model.xml";
    string strBinPath = strModelPath + "/saved_model.bin";
    pModel = new HFNetVINOModel(strXmlPath, strBinPath, mode, inputShape);
    if (pModel->IsValid())
    {
        cout << "Successfully loaded HFNet OpenVINO model."
             << " Mode: " << gStrModelDetectionName[mode]
             << " Shape: " << inputShape.t() << endl;
    } else
        exit(-1);

    return pModel;
}

HFNetBaseModel* InitModel(HFNetSettings* settings, ModelDetectionMode mode, cv::Vec4i inputShape) {
    HFNetBaseModel* pModel;
    if (settings->modelType() == kHFNetTFModel)
    {
        pModel = InitTFModel(settings->strModelPath(), mode, inputShape);
    } else if (settings->modelType() == kHFNetRTModel)
    {
        pModel = InitRTModel(settings->strModelPath(), mode, inputShape);
    } else if (settings->modelType() == kHFNetVINOModel)
    {
        string strModelPath;
        if (mode != kIntermediateToGlobal)
            strModelPath = settings->strModelPath() + "/local_part";
        else
            strModelPath = settings->strModelPath() + "/global_part";

        pModel = InitVINOModel(strModelPath, mode, inputShape);
    } else
    {
        cerr << "Wrong type of detector!" << endl;
        exit(-1);
    }

    return pModel;
}

void ExtractorNode::DivideNode(ExtractorNode& n1, ExtractorNode& n2, ExtractorNode& n3, ExtractorNode& n4) {
    const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
    const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

    // Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x + halfX, UL.y);
    n1.BL = cv::Point2i(UL.x, UL.y + halfY);
    n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x, UL.y + halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x, BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    // Associate points to childs
    for (size_t i = 0; i < vKeys.size(); i++)
    {
        const cv::KeyPoint& kp = vKeys[i];
        if (kp.pt.x < n1.UR.x)
        {
            if (kp.pt.y < n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        } else if (kp.pt.y < n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if (n1.vKeys.size() == 1)
        n1.bNoMore = true;
    if (n2.vKeys.size() == 1)
        n2.bNoMore = true;
    if (n3.vKeys.size() == 1)
        n3.bNoMore = true;
    if (n4.vKeys.size() == 1)
        n4.bNoMore = true;
}

static bool compareNodes(pair<int, ExtractorNode*>& e1, pair<int, ExtractorNode*>& e2) {
    if (e1.first < e2.first) {
        return true;
    } else if (e1.first > e2.first) {
        return false;
    } else {
        if (e1.second->UL.x < e2.second->UL.x) {
            return true;
        } else {
            return false;
        }
    }
}

std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int minX,
                                            const int maxX, const int minY, const int maxY, const int N) {
    // Compute how many initial nodes
    const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));

    const float hX = static_cast<float>(maxX - minX) / nIni;

    list<ExtractorNode> lNodes;

    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for (int i = 0; i < nIni; i++)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
        ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
        ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
        ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    // Associate points to childs
    for (size_t i = 0; i < vToDistributeKeys.size(); i++)
    {
        const cv::KeyPoint& kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();

    while (lit != lNodes.end())
    {
        if (lit->vKeys.size() == 1)
        {
            lit->bNoMore = true;
            lit++;
        } else if (lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    vector<pair<int, ExtractorNode*>> vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    while (!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while (lit != lNodes.end())
        {
            if (lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            } else
            {
                // If more than one point, subdivide
                ExtractorNode n1, n2, n3, n4;
                lit->DivideNode(n1, n2, n3, n4);

                // Add childs if they contain points
                if (n1.vKeys.size() > 0)
                {
                    lNodes.push_front(n1);
                    if (n1.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n2.vKeys.size() > 0)
                {
                    lNodes.push_front(n2);
                    if (n2.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n3.vKeys.size() > 0)
                {
                    lNodes.push_front(n3);
                    if (n3.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n4.vKeys.size() > 0)
                {
                    lNodes.push_front(n4);
                    if (n4.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit = lNodes.erase(lit);
                continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
        {
            bFinish = true;
        } else if (((int)lNodes.size() + nToExpand * 3) > N)
        {
            while (!bFinish)
            {
                prevSize = lNodes.size();

                vector<pair<int, ExtractorNode*>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end(), compareNodes);
                for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
                {
                    ExtractorNode n1, n2, n3, n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.vKeys.size() > 0)
                    {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0)
                    {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0)
                    {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0)
                    {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if ((int)lNodes.size() >= N)
                        break;
                }

                if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
                    bFinish = true;
            }
        }
    }

    // Retain the best point in each node
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(N);
    for (list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++)
    {
        vector<cv::KeyPoint>& vNodeKeys = lit->vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for (size_t k = 1; k < vNodeKeys.size(); k++)
        {
            if (vNodeKeys[k].response > maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

// copy from tensorflow.contrib.resampler
void Resampler(const float* data, const float* warp, float* output,
               const int batch_size, const int data_height,
               const int data_width, const int data_channels, const int num_sampling_points) {
    const int warp_batch_stride = num_sampling_points * 2;
    const int data_batch_stride = data_height * data_width * data_channels;
    const int output_batch_stride = num_sampling_points * data_channels;
    const float zero = static_cast<float>(0.0);
    const float one = static_cast<float>(1.0);

    auto resample_batches = [&](const int start, const int limit) {
        for (int batch_id = start; batch_id < limit; ++batch_id) {
            // Utility lambda to access data point and set output values.
            // The functions take care of performing the relevant pointer
            // arithmetics abstracting away the low level details in the
            // main loop over samples. Note that data is stored in NHWC format.
            auto set_output = [&](const int sample_id, const int channel,
                                  const float value) {
                output[batch_id * output_batch_stride + sample_id * data_channels +
                       channel] = value;
            };

            auto get_data_point = [&](const int x, const int y, const int chan) {
                const bool point_is_in_range =
                    (x >= 0 && y >= 0 && x <= data_width - 1 && y <= data_height - 1);
                return point_is_in_range
                           ? data[batch_id * data_batch_stride +
                                  data_channels * (y * data_width + x) + chan]
                           : zero;
            };

            for (int sample_id = 0; sample_id < num_sampling_points; ++sample_id) {
                const float x = warp[batch_id * warp_batch_stride + sample_id * 2];
                const float y = warp[batch_id * warp_batch_stride + sample_id * 2 + 1];
                // The interpolation function:
                // a) implicitly pads the input data with 0s (hence the unusual checks
                // with {x,y} > -1)
                // b) returns 0 when sampling outside the (padded) image.
                // The effect is that the sampled signal smoothly goes to 0 outside
                // the original input domain, rather than presenting a jump
                // discontinuity at the image boundaries.
                if (x > static_cast<float>(-1.0) && y > static_cast<float>(-1.0) &&
                    x < static_cast<float>(data_width) &&
                    y < static_cast<float>(data_height)) {
                    // Precompute floor (f) and ceil (c) values for x and y.
                    const int fx = std::floor(static_cast<float>(x));
                    const int fy = std::floor(static_cast<float>(y));
                    const int cx = fx + 1;
                    const int cy = fy + 1;
                    const float dx = static_cast<float>(cx) - x;
                    const float dy = static_cast<float>(cy) - y;

                    for (int chan = 0; chan < data_channels; ++chan) {
                        const float img_fxfy = dx * dy * get_data_point(fx, fy, chan);
                        const float img_cxcy =
                            (one - dx) * (one - dy) * get_data_point(cx, cy, chan);
                        const float img_fxcy = dx * (one - dy) * get_data_point(fx, cy, chan);
                        const float img_cxfy = (one - dx) * dy * get_data_point(cx, fy, chan);
                        set_output(sample_id, chan,
                                   img_fxfy + img_cxcy + img_fxcy + img_cxfy);
                    }
                } else {
                    for (int chan = 0; chan < data_channels; ++chan) {
                        set_output(sample_id, chan, zero);
                    }
                }
            }
        }
    };

    resample_batches(0, batch_size);
}

std::vector<cv::KeyPoint> NMS(const std::vector<cv::KeyPoint>& vToDistributeKeys, int width, int height, int radius) {
    std::vector<std::vector<const cv::KeyPoint*>> vpKeypoints(height, vector<const cv::KeyPoint*>(width, nullptr));
    unordered_set<const cv::KeyPoint*> selected;
    for (const auto& kpt : vToDistributeKeys)
    {
        vpKeypoints[kpt.pt.y][kpt.pt.x] = &kpt;
        selected.insert(&kpt);
    }

    for (const auto& kpt : vToDistributeKeys)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            for (int dy = -radius; dy <= radius; ++dy)
            {
                const int x = kpt.pt.x + dx;
                const int y = kpt.pt.y + dy;
                if (x < 0 || y < 0 || x >= width || y >= height) continue;
                if (!vpKeypoints[y][x]) continue;

                if (vpKeypoints[kpt.pt.y][kpt.pt.x]->response < vpKeypoints[y][x]->response)
                {
                    selected.erase(vpKeypoints[kpt.pt.y][kpt.pt.x]);
                    vpKeypoints[kpt.pt.y][kpt.pt.x] = nullptr;
                    goto jump;
                }
            }
        }
    jump:;
    }

    std::vector<cv::KeyPoint> res;
    res.reserve(selected.size());
    for (const auto& pKP : selected)
    {
        res.emplace_back(*pKP);
    }
    return res;
}

}  // namespace hfnet