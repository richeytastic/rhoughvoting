#include "Forest.h"
using HoughVoting::Forest;



void Forest::setFeatureExtractorSpecs( const vector<string>& specs, bool fixedPatch, cv::Size2f patchDims, bool offsetScaling)
{
    _fxspecs = specs;
    _fixedPatch = fixedPatch;
    _patchDims = patchDims;
    _offsetScaling = offsetScaling;
}   // end setFeatureExtractorSpecs



int Forest::getFeatureExtractorSpecs( vector<string>& specs) const
{
    specs.insert( specs.end(), _fxspecs.begin(), _fxspecs.end());
    return _fxspecs.size();
}   // end getFeatureExtractorSpecs
