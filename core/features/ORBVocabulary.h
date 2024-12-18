#pragma once

#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>

namespace slamplay {

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
    ORBVocabulary;

}  // namespace slamplay
