#include "pch.h"
#include "Oculus.h"
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <regex>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/photo.hpp"

//перевести на wstring


namespace tes = tesseract;

std::mutex mute;

void addBlackBorderToTheTopOfImage(cv::Mat& srcMat, cv::Mat& dstMat, int borderSz) {

	cv::Mat inpImgCuttedFromBottom(srcMat, cv::Rect(0, 0, srcMat.size().width, srcMat.size().height - borderSz));
	cv::Mat inpImgNew = cv::Mat::zeros(cv::Size(srcMat.size().width, srcMat.size().height), srcMat.type());;
	cv::Mat insetImage(inpImgNew, cv::Rect(0, borderSz, inpImgNew.size().width, inpImgNew.size().height - borderSz));
	inpImgCuttedFromBottom.copyTo(insetImage);
	dstMat = inpImgNew;

	return;
}

void tryFindPatternThreadProc(
	ThreadDescription *pThisThreadDescr
	,const std::string &imgFilePathName
	,const std::string &patternToFind
	,const std::string& ocrDataPath
	,const int rotationDegrees
	,std::vector<std::string>& errDescrVec
	,std::vector<std::string>& errLevelVec
	,int heightDivider
	,bool tryImproveQuality
	,bool resize) {

	int baseImgSz = 1080;
	int imgSz = 1024;
	int borderSize = 40;
	double ratio = 0;
	int bigSize = 0;

	cv::Mat inpImg;
	if (tryImproveQuality) {
		
		inpImg = cv::imread(imgFilePathName, cv::IMREAD_UNCHANGED && cv::IMREAD_IGNORE_ORIENTATION);
		if (inpImg.empty()) {
			pThisThreadDescr->isRunning = false;
			return;
		}
		
		bigSize = inpImg.size().height > inpImg.size().width ? inpImg.size().height : inpImg.size().width;

		if (bigSize > 1200) {
			ratio = (double)inpImg.size().height / (double)inpImg.size().width;
			/*std::lock_guard<std::mutex>* lg = new std::lock_guard<std::mutex>(mute);
			std::cout << "unch " << inpImg.size().width << "x" << inpImg.size().height << "=" << ratio << std::endl;
			delete lg;*/

			double div = (double)baseImgSz * ratio;
			int newHeight = (int)std::round(div);
			if (newHeight == 0) {
				pThisThreadDescr->isRunning = false;
				return;
			}
			cv::resize(inpImg, inpImg, cv::Size(baseImgSz, newHeight));
		}
		

		//add black border to the top of image (tesseract likes this (it's no joke))
		addBlackBorderToTheTopOfImage(inpImg, inpImg, borderSize);
		

		if (resize) {
			/*int newHeight = (int)imgSz * (int)ratio;
			if (newHeight == 0) {
				pThisThreadDescr->isRunning = false;
				return;
			}
			cv::resize(inpImg, inpImg, cv::Size(imgSz, newHeight));*/
			cv::resize(inpImg, inpImg, cv::Size(imgSz, imgSz));
		}

		if (inpImg.empty()) {
			errLevelVec.push_back("error");
			errDescrVec.push_back("Opencv can't read image.");
			pThisThreadDescr->isRunning = false;
			return;
			
		}
		cv::detailEnhance(inpImg, inpImg, 100, 1);
		cv::cvtColor(inpImg, inpImg, cv::COLOR_BGR2GRAY);
		cv::threshold(inpImg, inpImg, 50, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);	//...KA
	}
	else {
		inpImg = cv::imread(imgFilePathName, cv::IMREAD_GRAYSCALE && cv::IMREAD_IGNORE_ORIENTATION);
		if (inpImg.empty()) {
			pThisThreadDescr->isRunning = false;
			return;
		}

		bigSize = inpImg.size().height > inpImg.size().width ? inpImg.size().height : inpImg.size().width;

		if (bigSize > 1200) {
			ratio = (double)inpImg.size().height / (double)inpImg.size().width;
			/*std::lock_guard<std::mutex>* lg = new std::lock_guard<std::mutex>(mute);
			std::cout << "unch " << inpImg.size().width << "x" << inpImg.size().height << "=" << ratio << std::endl;
			delete lg;*/

			double div = (double)baseImgSz * ratio;
			int newHeight = (int)std::round(div);
			if (newHeight == 0) {
				pThisThreadDescr->isRunning = false;
				return;
			}
			cv::resize(inpImg, inpImg, cv::Size(baseImgSz, newHeight));
		}

		//add black border to the top of image (tesseract likes this (it's no joke))
		addBlackBorderToTheTopOfImage(inpImg, inpImg, borderSize);
		
		
		if (resize) {
			/*int newHeight = (int)baseImgSz * ratio;
			if (newHeight == 0) {
				pThisThreadDescr->isRunning = false;
				return;
			}
			cv::resize(inpImg, inpImg, cv::Size(imgSz, newHeight));*/
			cv::resize(inpImg, inpImg, cv::Size(imgSz, imgSz));
		}
		if (inpImg.empty()) {
			errLevelVec.push_back("error");
			errDescrVec.push_back("Opencv can't read image.");
			pThisThreadDescr->isRunning = false;
			return;
		}
	};

	tes::TessBaseAPI ocr;
	try
	{
		ocr.Init(ocrDataPath.c_str(), "rus");
	}
	catch (const std::exception& ex)
	{
		errLevelVec.push_back("error");
		std::string err("Tesseract init failure: ");
		err += ex.what();
		errDescrVec.push_back(err);
		pThisThreadDescr->isRunning = false;
		return;
	}
	if (rotationDegrees) {
		cv::Point2f pc(static_cast<float>(inpImg.cols) / 2, static_cast<float>(inpImg.rows) / 2);
		cv::Mat r = cv::getRotationMatrix2D(pc, rotationDegrees, 1.0);
		cv::warpAffine(inpImg, inpImg, r, cv::Size(inpImg.size().width, inpImg.size().height));
	}
	
	/*cv::imshow(std::to_string(rotationDegrees), inpImg);
	cv::waitKey();*/
	ocr.SetImage(static_cast<uchar*>(inpImg.data), inpImg.size().width, inpImg.size().height/heightDivider, inpImg.channels(), inpImg.step1());
	std::unique_ptr<char> tOut(ocr.GetUTF8Text());
	/*ocr.Clear();
	ocr.ClearPersistentCache();*/

	std::string sOut(tOut.get());
	sOut = std::regex_replace(sOut, std::regex("\\."), "");			//уберем точки и пробелы
	sOut = std::regex_replace(sOut, std::regex("\\s"), "");			//уберем точки и пробелы

	std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
	std::wstring wOut(converter.from_bytes(sOut));

	std::wstring wPattern(converter.from_bytes(patternToFind));

	if (wOut.find(wPattern, 0) != std::wstring::npos) {
		pThisThreadDescr->result = true;
	}

	pThisThreadDescr->isRunning = false;
	return;
}

int getActiveThreadsCount(std::vector<UPThreadDescription>* thVec) {
	int count = 0;

	for (std::vector<UPThreadDescription>::const_iterator it = thVec->cbegin(); it != thVec->cend(); std::advance(it, 1)) {
		if (it->get()->isRunning)
			count++;
	}

	return count;
}

void tryFindPattern(
	const std::string inPattern,
	bool& outResult,
	const std::string& imgFilePathName,
	const std::string& ocrDataPath,
	std::vector<std::string>& errDescr,
	std::vector<std::string>& errLevel,
	int maxThreads
	) {


	outResult = false;
	errDescr.clear();
	errLevel.clear();

	UPThreadDescriptionVector upThrVec = std::make_unique<std::vector<UPThreadDescription>>();
	

	int rotationDegrees = 0;
	for (int t = 0; t < 4; t++) {
		//int divider = 1;
		for (int divider = 1; divider < 11; divider++) {
			/*int improve = 0;
			int resize = 0; */
			for (int improve = 0; improve < 2; improve++) {
				for (int resize = 0; resize < 2; resize++) {
					
					while (getActiveThreadsCount(upThrVec.get()) >= maxThreads) {
						std::this_thread::sleep_for(std::chrono::milliseconds(1000));
					}

					UPThreadDescription upTd = std::make_unique<ThreadDescription>();
					UPThread upThread = std::make_unique<std::thread>(
						tryFindPatternThreadProc
						, upTd.get()
						, std::cref(imgFilePathName)
						, std::cref(inPattern)
						, std::cref(ocrDataPath)
						, rotationDegrees
						, std::ref(errDescr)
						, std::ref(errLevel)
						, divider
						, improve
						, resize);

					upTd->pThread = std::move(upThread);
					upTd->threadId = upTd->pThread->get_id();
					std::thread* pThreadLoc = upTd->pThread.get();
					upThrVec->push_back(std::move(upTd));
					pThreadLoc->detach();
				}
			}
		}
		rotationDegrees += 90;
	}

	std::this_thread::sleep_for(std::chrono::seconds(1));

	bool threadsExecuting = true;
	while (threadsExecuting) {
		threadsExecuting = false;

		for (auto &upTd: (*upThrVec.get())) {
			threadsExecuting = threadsExecuting || upTd->isRunning;
			outResult = outResult || upTd->result;
		}
	}

	return;
}