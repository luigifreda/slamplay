#include "SamInterface.h"

#include "io/file_utils.h"
#include "sam_tensorrt/Sam.h"
#include "sam_tensorrt/sam_export.h"
#include "sam_tensorrt/sam_utils.h"

#include <macros.h>

const std::string kDataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag
const std::string kSamDataDir = kDataDir + "/segment_anything";

///////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;

int main(int argc, char const* argv[]) {
    const std::string inputImage = kSamDataDir + "/truck-vga.png";

    cv::Mat frame = cv::imread(inputImage);
    std::cout << "frame size:" << frame.size << std::endl;

    SamInterface samIf;
    samIf.loadModels();

    samIf.processEmbedding(frame);
    cv::Mat mask = samIf.processAutoSegment(frame);

    cv::Mat outImage;
    samIf.showAutoSegmentResult(frame, mask, outImage);
    cv::imshow("auto segmentation", outImage);

    cv::waitKey(0);
}