﻿#include <iostream>
#include <string>
#include <vector>
#include "Test.h"
#include "Oculus.h"


int main(int argc, const char** argv)
{
    std::string imgFileName;
    std::string patternToFind;
    std::string ocrDataPath;

    for (int i = 0; i != argc; i++) {
        if (!strcmp(argv[i], "-img")) {
            i++;
            if (i >= argc)
                break;
            imgFileName.assign(argv[i]);
        }

        if (!strcmp(argv[i], "-pat")) {
            i++;
            if (i >= argc)
                break;
            patternToFind.assign(argv[i]);
        }

        if (!strcmp(argv[i], "-data")) {
            i++;
            if (i >= argc)
                break;
            ocrDataPath.assign(argv[i]);
        }
    }

    if (!imgFileName.length()) {
        std::cout << "***ERROR*** " << "No pass to img was specifyed." << std::endl;
        return -1;
    }

    if (!ocrDataPath.length()) {
        std::cout << "***ERROR*** " << "No pass to trained data was specifyed." << std::endl;
        return -1;
    }

    if (!patternToFind.size()) {
        std::cout << "***ERROR*** " << "No pattern to find was specifyed." << std::endl;
        return -1;
    }

    std::vector<std::string> errDescr;
    std::vector<std::string> errLevel;
    bool result = false;

    tryFindPattern(patternToFind, result, imgFileName, ocrDataPath, errDescr, errLevel, 4);
    std::cout << imgFileName << " status:" << result << std::endl;


    return result;
}
