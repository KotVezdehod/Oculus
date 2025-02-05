﻿#include <iostream>
#include <string>
#include <vector>
#include "Test.h"
#include "Oculus.h"

int main()
{
    
    std::string imgFilePathName;
    std::string ocrDataPath;
    std::string dbgResult;
    bool result = false;

    imgFilePathName.assign("C:\\C++\\Oculus\\010824_0.jpg");
    ocrDataPath.assign("C:\\C++\\Oculus");
    tryFindPattern("918682", result, imgFilePathName, ocrDataPath, dbgResult, 8);
    std::cout << imgFilePathName << " status:" << result << std::endl;

    imgFilePathName.assign("C:\\C++\\Oculus\\010824_1.jpg");
    ocrDataPath.assign("C:\\C++\\Oculus");
    tryFindPattern("1008326", result, imgFilePathName, ocrDataPath, dbgResult, 8);
    std::cout << imgFilePathName << " status:" << result << std::endl;

    imgFilePathName.assign("C:\\C++\\Oculus\\v8_6583_3a0c.png");
    ocrDataPath.assign("C:\\C++\\Oculus");
    tryFindPattern("2004380", result, imgFilePathName, ocrDataPath, dbgResult, 8);
    std::cout << imgFilePathName << " status:" << result << std::endl;
        
    imgFilePathName.assign("C:\\C++\\Oculus\\2.jpg");
    ocrDataPath.assign("C:\\C++\\Oculus");
    tryFindPattern("878543", result, imgFilePathName, ocrDataPath, dbgResult, 8);
    std::cout << imgFilePathName << " status:" << result << std::endl;

    imgFilePathName.assign("C:\\C++\\Oculus\\notRect_orig.jpg");
    ocrDataPath.assign("C:\\C++\\Oculus");
    tryFindPattern("2672021", result, imgFilePathName, ocrDataPath, dbgResult, 8);
    std::cout << imgFilePathName << " status:" << result << std::endl;

    imgFilePathName.assign("C:\\C++\\Oculus\\Oktiabr.jpg");
    ocrDataPath.assign("C:\\C++\\Oculus");
    tryFindPattern("2624243", result, imgFilePathName, ocrDataPath, dbgResult, 8);
    std::cout << imgFilePathName << " status:" << result << std::endl;


    imgFilePathName.assign("C:\\C++\\Oculus\\v8_6583_3a00.png");
    ocrDataPath.assign("C:\\C++\\Oculus");
    tryFindPattern("2672245", result, imgFilePathName, ocrDataPath, dbgResult, 8);
    std::cout << imgFilePathName << " status:" << result << std::endl;


    imgFilePathName.assign("C:\\C++\\Oculus\\v8_6583_3a0a.png");
    ocrDataPath.assign("C:\\C++\\Oculus");
    tryFindPattern("2629757", result, imgFilePathName, ocrDataPath, dbgResult, 8);
    std::cout << imgFilePathName << " status:" << result << std::endl;


    imgFilePathName.assign("C:\\C++\\Oculus\\v8_6583_3a1c.png");
    ocrDataPath.assign("C:\\C++\\Oculus");
    tryFindPattern("2629758", result, imgFilePathName, ocrDataPath, dbgResult, 8);
    std::cout << imgFilePathName << " status:" << result << std::endl;


    imgFilePathName.assign("C:\\C++\\Oculus\\v8_6583_3a1e.png");
    ocrDataPath.assign("C:\\C++\\Oculus");
    tryFindPattern("1127228", result, imgFilePathName, ocrDataPath, dbgResult, 8);
    std::cout << imgFilePathName << " status:" << result << std::endl;


    imgFilePathName.assign("C:\\C++\\Oculus\\v8_6583_3a74.png");
    ocrDataPath.assign("C:\\C++\\Oculus");
    tryFindPattern("2004414", result, imgFilePathName, ocrDataPath, dbgResult, 8);
    std::cout << imgFilePathName << " status:" << result << std::endl;


    imgFilePathName.assign("C:\\C++\\Oculus\\v8_D5BE_4bc1.png");
    ocrDataPath.assign("C:\\C++\\Oculus");
    tryFindPattern("1990971", result, imgFilePathName, ocrDataPath, dbgResult, 8);
    std::cout << imgFilePathName << " status:" << result << std::endl;


    imgFilePathName.assign("C:\\C++\\Oculus\\v8_D5BE_4d53.png");
    ocrDataPath.assign("C:\\C++\\Oculus");
    tryFindPattern("2615290", result, imgFilePathName, ocrDataPath, dbgResult, 8);
    std::cout << imgFilePathName << " status:" << result << std::endl;

    

}

