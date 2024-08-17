#include "SamInterface.h"

#include "io/file_utils.h"
#include "sam_tensorrt/Sam.h"
#include "sam_tensorrt/sam_export.h"
#include "sam_tensorrt/sam_utils.h"

#include <macros.h>

const std::string kDataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag
const std::string kSamDataDir = kDataDir + "/segment_anything";
const std::string kSamModelsDir = kSamDataDir + "/models";

///////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;
using namespace slamplay;

void segmentClickedPoint(const cv::Mat& image, SamInterface& samIf) {
    // Data structure to hold clicked point
    PointData pointData;
    pointData.clicked = false;
    cv::Mat clonedImage = image.clone();
    int clickCount = 0;

    bool segmentAll = false;

    // Loop until Esc key is pressed
    while (true)
    {
        // Display the original image
        cv::imshow("Image", clonedImage);

        if (segmentAll) {
            segmentAll = false;
        } else if (pointData.clicked)
        {
            // reset the image before prediction
            clonedImage = image.clone();

            std::cout << "got new click " << pointData.point << std::endl;
            pointData.clicked = false;  // Reset clicked flag

            cv::Mat mask = samIf.processSinglePointMask(image, pointData.point.x, pointData.point.y);
            cv::circle(clonedImage, pointData.point, 5, cv::Scalar(0, 0, 255), -1);

            if (clickCount >= CITYSCAPES_COLORS.size()) clickCount = 0;
            overlay(clonedImage, mask, CITYSCAPES_COLORS[clickCount]);
            clickCount++;
        }

        // Set the callback function for mouse events on the displayed cloned image
        cv::setMouseCallback("Image", onMouse, &pointData);

        // Check for Esc key press
        char key = cv::waitKey(1);
        if (key == 27)  // ASCII code for Esc key
        {
            break;
        } else if (key == 'c' || key == 'C')
        {
            std::cout << "clear all" << std::endl;
            clonedImage = image.clone();
            clickCount = 0;
        } else if (key == 'a' || key == 'A')
        {
        }
    }
    cv::destroyAllWindows();
}

int main(int argc, char const* argv[]) {
    std::string inputImage = kSamDataDir + "/truck-vga.png";

    cv::Mat frame = cv::imread(inputImage);
    std::cout << "frame size:" << frame.size << std::endl;

    SamInterface samIf;
    samIf.loadModels();

    auto& image_embeddings = samIf.processEmbedding(frame);

#if 0
        // basic test
        auto res = samIf.engPromptEncAndMaskDec->prepareInput(100, 100, frame, image_embeddings);
        // std::vector<int> mult_pts = {x,y,x-5,y-5,x+5,y+5};
        // auto res = engPromptEncAndMaskDec->prepareInput(mult_pts, image_embeddings);
        std::cout << "------------------prepareInput: " << res << std::endl;
        res = samIf.engPromptEncAndMaskDec->infer();
        std::cout << "------------------infer: " << res << std::endl;
        samIf.engPromptEncAndMaskDec->verifyOutput();
        std::cout << "-----------------done" << std::endl;
#else
    segmentClickedPoint(frame, samIf);
#endif
}
