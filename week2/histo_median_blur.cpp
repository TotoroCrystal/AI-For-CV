#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

//计算亮度中值和灰度<=中值的像素点个数
void CalculateImage_MedianGray_PixelCount(const Mat &histogram, int pixelCount, int &medianValue, int &pixleCountLowerMedian)
{
    float *data = (float *)histogram.data;//直方图
    int sum = 0;
    for (int i = 0; i <= 255; ++i)
    {
        //
        sum += data[i];
        if (2 * sum>pixelCount)
        {
            medianValue = i;
            pixleCountLowerMedian = sum;
            break;
        }
    }
}

void fastMedianBlur(const Mat &srcImg, Mat &dstImg, int diameter)
{
    int radius = (diameter - 1) / 2;
    int imgW = srcImg.cols;
    int imgH = srcImg.rows;
    int channels = srcImg.channels();
    dstImg = Mat::zeros(imgH, imgW, CV_8UC1);
    int windowSize = diameter*diameter;
    Mat windowImg = Mat::zeros(diameter, diameter, CV_8UC1);
    
    //直方图
    Mat histogram;
    int histogramSize = 256;//灰度等级
    int thresholdValue = windowSize / 2 + 1;//step1.设置阈值(步骤参考：图像的高效编程要点之四)
    
    //待处理图像像素
    uchar *pDstData = dstImg.data + imgW*radius + radius;
    //整个图像中窗口位置指针
    uchar *pSrcData = srcImg.data;
    
    //逐行遍历
    for (int i = radius; i <= imgH - 1 - radius; i++)
    {
        //列指针
        uchar *pColDstData = pDstData;
        uchar *pColSrcData = pSrcData;
        
        //单个窗口指针
        uchar *pWindowData = windowImg.data;
        uchar *pRowSrcData = pColSrcData;
        //从源图中提取窗口图像
        for (int winy = 0; winy <= diameter - 1; winy++)
        {
            for (int winx = 0; winx <= diameter - 1; winx++)
            {
                pWindowData[winx] = pRowSrcData[winx];
            }
            pRowSrcData += imgW;
            pWindowData += diameter;
        }
        
        //求直方图,确定中值，并记录亮度<=中值的像素点个数
        calcHist(&windowImg,
                 1,//Mat的个数
                 0,//用来计算直方图的通道索引，通道索引依次排开
                 Mat(),//Mat()返回一个空值，表示不用mask,
                 histogram, //直方图
                 1, //直方图的维数，如果计算2个直方图，就为2
                 &histogramSize, //直方图的等级数(如灰度等级),也就是每列的行数
                 0//分量的变化范围
                 );
        
        //求亮度中值和<=中值的像素点个数
        int medianValue, pixelCountLowerMedian;
        CalculateImage_MedianGray_PixelCount(histogram, windowSize, medianValue, pixelCountLowerMedian);
        ////////////滑动窗口操作结束///////////////////////
        
        //滤波
        pColDstData[0] = (uchar)medianValue;
        
        //处理同一行下一个像素
        pColDstData++;
        pColSrcData++;
        for (int j = radius + 1; j <= imgW - radius - 1; j++)
        {
            //维护滑动窗口直方图
            //删除左侧
            uchar *pWinLeftData = pColSrcData - 1;
            float *pHistData = (float*)histogram.data;
            for (int winy = 0; winy < diameter; winy++)
            {
                uchar grayValue = pWinLeftData[0];
                pHistData[grayValue] -= 1.0;
                if (grayValue <= medianValue)
                {
                    pixelCountLowerMedian--;
                }
                pWinLeftData += imgW;
            }
            
            //增加右侧
            uchar *pWinRightData = pColSrcData + diameter - 1;
            for (int winy = 0; winy < diameter; winy++)
            {
                uchar grayValue = pWinRightData[0];
                pHistData[grayValue] += 1.0;
                if (grayValue <= medianValue)
                {
                    pixelCountLowerMedian++;
                }
                pWinRightData += imgW;
            }
            //计算新的中值
            if (pixelCountLowerMedian > thresholdValue)
            {
                while (1)
                {
                    pixelCountLowerMedian -= pHistData[medianValue];
                    medianValue--;
                    if (pixelCountLowerMedian <= thresholdValue)
                    {
                        break;
                    }
                }
            }
            else
            {
                while (pixelCountLowerMedian < thresholdValue)
                {
                    medianValue++;
                    pixelCountLowerMedian += pHistData[medianValue];
                    
                }
            }
            
            pColDstData[0] = medianValue;
            //下一个像素
            pColDstData++;
            pColSrcData++;
        }
        //移动至下一行
        pDstData += imgW;
        pSrcData += imgW;
    }
    
    //边界直接赋原始值，不做滤波处理
    pSrcData = srcImg.data;
    pDstData = dstImg.data;
    //上下边界
    for (int j = 0; j < imgW; j++)
    {
        for (int i = 0; i < radius; i++)
        {
            int idxTop = i*imgW + j;
            pDstData[idxTop] = pSrcData[idxTop];
            int idxBot = (imgH - i - 1)*imgW + j;
            pDstData[idxBot] = pSrcData[idxBot];
        }
    }
    //左右边界
    for (int i = radius; i < imgH - radius - 1; i++)
    {
        for (int j = 0; j < radius; j++)
        {
            int idxLeft = i*imgW + j;
            pDstData[idxLeft] = pSrcData[idxLeft];
            int idxRight = i*imgW + imgW - j-1;
            pDstData[idxRight] = pSrcData[idxRight];
        }
    }
}


void main()
{
    string imgPath = "data/";
    Mat srcImg = imread(imgPath + "lenna.jpg", 0);
    Mat dstImg;
    double t0 = cv::getTickCount();
    fastMedianBlur(srcImg, dstImg, 5);
    //cv::medianBlur(srcImg, dstImg, 5); //OpenCV
    double t1 = cv::getTickCount();
    cout << "time=" << (t1 - t0) / cv::getTickFrequency() << endl;
    
    imwrite("data/test/srcImg.bmp", srcImg);
    imwrite("data/test/myFilter.bmp", dstImg);
}
