#pragma once
#include <string>
#include <iostream>
#include <memory>
#include <codecvt>
#include <fstream>
#include <thread>
#include <vector>

using StringVector = std::vector<std::string>;

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
	std::string &diagOut,
	int maxThreads = 4);

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
using UPThreadDescription = std::unique_ptr<ThreadDescription>;							//����� ����� ������� ����� using (� ���� ����� ��� template<typename T> ����������� � ����� - �����)))
typedef std::unique_ptr<std::vector<UPThreadDescription>> UPThreadDescriptionVector;	//��� ��������


void tryFindPatternThreadProc(
	ThreadDescription* pThisThreadDescr
	, const std::string& imgFilePathName
	, const std::string& patternToFind
	, const std::string& ocrDataPath
	, const int rotationDegrees
	, int heightDivider
	, bool tryImproveQuality
	, bool resize
	, StringVector* diagVecIn
	, int detectTopNoise);
