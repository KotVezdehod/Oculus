#pragma once
#include <string>
#include <iostream>
#include <memory>
#include <codecvt>
#include <fstream>
#include <thread>
#include <vector>


#ifdef OCULUS_EXPORTS 
	#define OCULUS_API __declspec(dllexport)
#else
	#define OCULUS_API __declspec(dllimport)
#endif // OCULUS_EXPORTS 

extern "C" OCULUS_API void tryFindPattern(
	const std::string inPattern,
	bool& outResult, 
	const std::string & imgFilePathName,
	const std::string &ocrDataPath,
	std::vector<std::string> &errDescr,
	std::vector<std::string> &errLevel);

typedef std::unique_ptr<std::thread> UPThread;

struct ThreadDescription
{
	std::thread::id threadId;
	UPThread pThread = nullptr;
	bool isRunning = true;
	bool result = false;

	//~ThreadDescription() {
	//	std::cout << "~ThreadDescription";
	//}
};

//template<typename T>
using UPThreadDescription = std::unique_ptr<ThreadDescription>;							//нашел такой вариант через using (к нему можно еще template<typename T> пристегнуть и будет - огонь)))
typedef std::unique_ptr<std::vector<UPThreadDescription>> UPThreadDescriptionVector;	//это классика



void tryFindPatternThreadProc(
	ThreadDescription* pThisThreadDescr
	, const std::string& imgFilePathName
	, const std::string& patternToFind
	, const std::string& ocrDataPath
	, const int rotationDegrees
	, std::vector<std::string>& errDescrVec
	, std::vector<std::string>& errLevelVec
	, int heightDivider
	, bool tryImproveQuality
	, bool resize);
