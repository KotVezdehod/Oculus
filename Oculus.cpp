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
using LockGuard = std::lock_guard<std::mutex>;

void improoveBrightness(cv::Mat &matIn) {
	cv::Mat new_image = cv::Mat::zeros(matIn.size(), matIn.type());

	double alpha = 1.4; /*< Simple contrast control [1,0-3,0] */
	double beta = 30;       /*< Simple brightness control [0-100] */

	for (int y = 0; y < matIn.rows; y++) {
		for (int x = 0; x < matIn.cols; x++) {
			for (int c = 0; c < matIn.channels(); c++) {
				try
				{
					new_image.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(alpha * matIn.at<cv::Vec3b>(y, x)[c] + beta);
				}
				catch (const std::exception&)
				{
					return;
				}
			}
		}
	}
	matIn = new_image;
}

void addBlackBorderToTheTopOfImage(cv::Mat& srcMat, cv::Mat& dstMat, int borderSz) {

	//исходное изо, за вычетом области сверху размером == размеру добаляемого прямоугольника
	cv::Mat inpImgCuttedFromBottom(srcMat, cv::Rect(0, borderSz, srcMat.size().width, srcMat.size().height - borderSz));
	
	//черный лист - с размерами как srcMat (в него будем внедрять inpImgCuttedFromBottom)
	cv::Mat blackBorderImg = cv::Mat::zeros(cv::Size(srcMat.size().width, srcMat.size().height), srcMat.type());		

	//область на черном листе, которая идет ниже нужной черной полосы и высота которой == (высота srcMat - высота черной полосы)
	cv::Mat insetImage(blackBorderImg, cv::Rect(0, borderSz, blackBorderImg.size().width, blackBorderImg.size().height - borderSz));
	inpImgCuttedFromBottom.copyTo(insetImage);
	dstMat = blackBorderImg;

	return;
}

double getWholeChanBrightnessPerPoint(cv::Mat& matIn) {
	double out = 0;
	double pointsCount = static_cast<double>(matIn.size().width) * static_cast<double>(matIn.size().height);

	cv::Scalar sc = cv::sum(matIn);
	
	for (UINT i = 0; i != matIn.channels(); i++) {
		out = out + sc[i];
	}

	out = out / pointsCount;

	return out;
}

int blackWaterTopNoise(cv::Mat& imgIn, bool tryImproveQuality) {

	UINT probeRect = 40;
	double sensitivityPerChennel = 8;
	double wholeSensitivity = sensitivityPerChennel * static_cast<double>(imgIn.channels());

	double brMid = getWholeChanBrightnessPerPoint(imgIn);

	UINT imgWidth = imgIn.size().width;
	UINT imgHeight = imgIn.size().height;
	UINT x_mid = imgWidth / 2;
	UINT y_mid = imgHeight / 2;

	UINT ySheetStart = 0;

	for (UINT curY = 0; curY < y_mid; curY += probeRect) {

		if (curY >= imgHeight / 3)
			break;

		UINT xStart = x_mid - (probeRect / 2);
		UINT xEnd = x_mid + (probeRect / 2);
		UINT yStart = curY;
		UINT yEnd = curY + probeRect;
		cv::Mat curRect(imgIn, cv::Rect(xStart, yStart, xEnd, yEnd));
		double brCurr = getWholeChanBrightnessPerPoint(curRect);

		UINT xnStart = x_mid - (probeRect / 2);
		UINT xnEnd = x_mid + (probeRect / 2);
		UINT ynStart = curY + probeRect;
		UINT ynEnd = curY + (probeRect * 2);
		cv::Mat nextRect(imgIn, cv::Rect(xStart, yStart, xEnd, yEnd));
		double brNxt = getWholeChanBrightnessPerPoint(nextRect);

		double deltaCurr = brCurr - brMid;
		double deltaNext = brNxt - brMid;

		if (deltaCurr >= wholeSensitivity 
			&& deltaNext >= wholeSensitivity) {
			ySheetStart = curY;
			break;
		};
	}
	if (ySheetStart > 0 && ySheetStart < imgHeight / 2) {

		if (tryImproveQuality)
			improoveBrightness(imgIn);

		addBlackBorderToTheTopOfImage(imgIn, imgIn, ySheetStart);
				
		return ySheetStart;
	};

	return 0;
}

void tryFindPatternThreadProc(
	ThreadDescription *pThisThreadDescr
	,const std::string &imgFilePathName
	,const std::string &patternToFind
	,const std::string& ocrDataPath
	,const int rotationDegrees
	,int heightDivider
	,bool tryImproveQuality
	,bool resize
	,StringVector* diagVecIn
	,int detectTopNoise) {

	int maxSize = 1200;
	int baseImgSz = 1080;
	//int imgSz = 1024;
	int borderSize = 40;
	double ratio = 0;
	int bigSize = 0;

	std::stringstream diagMsg;

	diagMsg << "OCULUS:: pattern: " << patternToFind << std::endl;
	diagMsg << "OCULUS:: rotationDegrees: " << rotationDegrees << std::endl;
	diagMsg << "OCULUS:: heightDivider: " << heightDivider << std::endl;
	diagMsg << "OCULUS:: tryImproveQuality: " << tryImproveQuality << std::endl;
	diagMsg << "OCULUS:: resize: " << resize << std::endl;
	diagMsg << "OCULUS:: detectTopNoise: " << detectTopNoise << std::endl;
	

	cv::Mat inpImg;
	if (tryImproveQuality) {
		
		inpImg = cv::imread(imgFilePathName, cv::IMREAD_UNCHANGED && cv::IMREAD_IGNORE_ORIENTATION);
		if (inpImg.empty()) {
			pThisThreadDescr->isRunning = false;
			return;
		}
		
		int bwEnd = 0;
		if (detectTopNoise > 0) {
			bwEnd = blackWaterTopNoise(inpImg, true);
			if (bwEnd) {
				diagMsg << "OCULUS:: Noise detected at the top of image. Adding blackwater down to " << bwEnd << std::endl;
			}
		}

		bigSize = inpImg.size().height > inpImg.size().width ? inpImg.size().height : inpImg.size().width;

		if (bigSize > maxSize) {
			diagMsg << "OCULUS:: Biggest dimension greater then " << maxSize << " resizing down to " << baseImgSz << std::endl;
			ratio = (double)inpImg.size().height / (double)inpImg.size().width;

			double div = (double)baseImgSz * ratio;
			int newHeight = (int)std::round(div);
			if (newHeight == 0) {
				pThisThreadDescr->isRunning = false;
				return;
			}
			cv::resize(inpImg, inpImg, cv::Size(baseImgSz, newHeight));
		}
		
		if (inpImg.empty()) {
			pThisThreadDescr->isRunning = false;
			return;

		};
		if (!bwEnd) {
			improoveBrightness(inpImg);
		};
		

		//cv::detailEnhance(inpImg, inpImg, 100, 1);
		//cv::cvtColor(inpImg, inpImg, cv::COLOR_BGR2GRAY);
		//cv::threshold(inpImg, inpImg, 50, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);	//...KA
	}
	else {
		
		inpImg = cv::imread(imgFilePathName, cv::IMREAD_GRAYSCALE && cv::IMREAD_IGNORE_ORIENTATION);
		if (inpImg.empty()) {
			pThisThreadDescr->isRunning = false;
			return;
		}
		
		int bwEnd = 0;
		if (detectTopNoise > 0) {
			bwEnd = blackWaterTopNoise(inpImg, false);
			if (bwEnd) {
				diagMsg << "OCULUS:: Noise detected at the top of image. Adding blackwater down to " << bwEnd << std::endl;
			}
		}

		bigSize = inpImg.size().height > inpImg.size().width ? inpImg.size().height : inpImg.size().width;

		if (bigSize > maxSize) {
			diagMsg << "OCULUS:: Biggest dimension greater then " << maxSize << " resizing down to " << baseImgSz << std::endl;
			ratio = (double)inpImg.size().height / (double)inpImg.size().width;

			double div = (double)baseImgSz * ratio;
			int newHeight = (int)std::round(div);
			if (newHeight == 0) {
				pThisThreadDescr->isRunning = false;
				return;
			}
			cv::resize(inpImg, inpImg, cv::Size(baseImgSz, newHeight));
		}

		if (inpImg.empty()) {
			pThisThreadDescr->isRunning = false;
			return;
		}

	};

	tes::TessBaseAPI ocr;
	try
	{
	
		ocr.Init(ocrDataPath.c_str(), "rus+eng");
		//ocr.SetVariable("tessedit_char_whitelist", "0123456789");
	}
	catch (const std::exception& )
	{
		pThisThreadDescr->isRunning = false;
		return;
	}
	if (rotationDegrees) {
		cv::Point2f pc(static_cast<float>(inpImg.cols) / 2, static_cast<float>(inpImg.rows) / 2);
		cv::Mat r = cv::getRotationMatrix2D(pc, rotationDegrees, 1.0);
		cv::warpAffine(inpImg, inpImg, r, cv::Size(inpImg.size().width, inpImg.size().height));
	}

	/*cv::imshow("1", inpImg);
	cv::waitKey();
	cv::destroyWindow("1");*/
	
	ocr.SetImage(static_cast<uchar*>(inpImg.data), inpImg.size().width, inpImg.size().height/heightDivider, inpImg.channels(), static_cast<int>(inpImg.step1()));
	std::unique_ptr<char> tOut(ocr.GetUTF8Text());
	
	std::string sOut(tOut.get());
	sOut = std::regex_replace(sOut, std::regex("\\."), "");			//уберем точки и пробелы
	sOut = std::regex_replace(sOut, std::regex("\\s"), "");			//уберем точки и пробелы

	std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
	std::wstring wOut(converter.from_bytes(sOut));

	std::wstring wPattern(converter.from_bytes(patternToFind));

	if (wOut.find(wPattern, 0) != std::wstring::npos) {
		pThisThreadDescr->result = true;

		std::unique_ptr<LockGuard> lg = std::make_unique<LockGuard>(mute);
		diagVecIn->push_back(diagMsg.str());
		lg.reset();
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
	std::string& diagOut,
	int maxThreads) {

	diagOut.clear();

	std::stringstream ssDiag;

	std::cout << "========== Image: " << imgFilePathName << "==========" << std::endl;
	ssDiag << "========== Image: " << imgFilePathName << "==========" << std::endl;
	
	outResult = false;
	std::unique_ptr<StringVector> sDiag = std::make_unique<StringVector>();

	UPThreadDescriptionVector upThrVec = std::make_unique<std::vector<UPThreadDescription>>();
	
	int divider = 1;
	int improve = 0;
	int resize = 0; 
	bool detected = false;
	int rotationDegrees = 0;
	int detectTopNoise = 0;
	for (int t = 0; t < 4; t++) {
		for (int divider = 1; divider < 11; divider++) {

			for (int improve = 0; improve < 2; improve++) {
				for (detectTopNoise = 0; detectTopNoise != 1; detectTopNoise++) {
					while (getActiveThreadsCount(upThrVec.get()) >= maxThreads) {
						std::this_thread::sleep_for(std::chrono::milliseconds(1000));
					}

					for (std::vector<UPThreadDescription>::const_iterator it0 = upThrVec->cbegin(); it0 != upThrVec->cend(); std::advance(it0, 1)) {
						if (it0->get()->result) {
							detected = true;
							break;
						}
					}
					if (detected)
						break;


					UPThreadDescription upTd = std::make_unique<ThreadDescription>();
					UPThread upThread = std::make_unique<std::thread>(
						tryFindPatternThreadProc
						, upTd.get()
						, std::cref(imgFilePathName)
						, std::cref(inPattern)
						, std::cref(ocrDataPath)
						, rotationDegrees
						, divider
						, improve
						, resize
						, sDiag.get()
						, detectTopNoise);

					upTd->pThread = std::move(upThread);
					upTd->threadId = upTd->pThread->get_id();
					std::thread* pThreadLoc = upTd->pThread.get();
					upThrVec->push_back(std::move(upTd));
					pThreadLoc->detach();
				}
				if (detected)
					break;
			}
			if (detected)
				break;
		}
		if (detected)
			break;
		rotationDegrees += 90;
	};

	std::this_thread::sleep_for(std::chrono::seconds(1));

	bool threadsExecuting = true;
	while (threadsExecuting) {
		threadsExecuting = false;

		for (auto &upTd: (*upThrVec.get())) {
			threadsExecuting = threadsExecuting || upTd->isRunning;
			outResult = outResult || upTd->result;
		}
	}
		
	if (outResult && sDiag->size()) {
		
		std::cout << sDiag->at(0);
		ssDiag << sDiag->at(0);
	}

	std::cout << "OCULUS:: Result " << outResult << std::endl;

	ssDiag << "OCULUS:: Result " << outResult << std::endl;
	diagOut = ssDiag.str();


	return;
}