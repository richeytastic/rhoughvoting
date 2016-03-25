#include "TestParamGenerator.h"
using HoughVoting::TestParamGenerator;
#include <cassert>
#include <cstdlib>


TestParamGenerator::TestParamGenerator( int numFVs, int* fvRanges, int maxTests)
    : _maxTests(maxTests), _testIdx(0)
{
    _fvRanges.insert(_fvRanges.end(), fvRanges, &fvRanges[numFVs]); // End pointer is one past the end
    _fvId = 0;
    _i0.insert( _i0.end(), numFVs, 0);
    _i1.insert( _i1.end(), numFVs, 0);
    int numPoss = 0;    // Find the number of possible parameter settings
    for ( int i = 0; i < numFVs; ++i)
    {
        assert( _fvRanges[i] >= 2);
        numPoss += (_fvRanges[i] * (_fvRanges[1]-1))/2;
    }   // end for
    _doRandom = numPoss > maxTests;
}   // end ctor


int TestParamGenerator::nextParams( int& i0, int& i1)
{
    int testFV = 0;
    const bool rv = _doRandom ? generateRandom( testFV, i0, i1) : generateNext( testFV, i0, i1);
    _testIdx++;
    if ( !rv) testFV = _fvRanges.size();
    return testFV;
}   // end nextParams


bool TestParamGenerator::generateRandom( int& testFV, int& i0, int& i1) const
{
    if ( _testIdx >= _maxTests)
        return false;
    testFV = random() % _fvRanges.size();    // Choose a random FV
    i0 = random() % _fvRanges[testFV];  // Choose a random index into the FV
    while (( i1 = random() % _fvRanges[testFV]) == i0);  // Choose a random index into the FV that isn't i0
    return true;
}   // end generateRandom


bool TestParamGenerator::generateNext( int& testFV, int& i0, int& i1)
{
    if ( _fvId >= _fvRanges.size())
        return false;

    _i1[_fvId]++;

    if ( _i1[_fvId] == _fvRanges[_fvId])
    {
        _i0[_fvId]++;

        if ( _i0[_fvId] == _fvRanges[_fvId]-1)
        {
            _fvId++;    // Exhausted
            if ( _fvId >= _fvRanges.size())
                return false;

            _i0[_fvId] = 0;
        }   // end if

        _i1[_fvId] = _i0[_fvId]+1;
    }   // end if

    testFV = _fvId;
    i0 = _i0[_fvId];
    i1 = _i1[_fvId];

    return true;
}   // end generateNext
