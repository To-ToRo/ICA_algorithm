#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*
 * The number of channel and the value of alpha
 * will be determined at hardware level design with Verilog.
 */
#define CH_NUM 16
#define ALPHA 1.2
#define MAX_TIME 10000
#define LEARN_COUNT 100

double sigmoid(double x)
{
    return (1 / (1 + exp(-x)));
}

double sample_data[CH_NUM][MAX_TIME];
double source_data[CH_NUM][MAX_TIME];

int main(void)
{
    /*
     * Generate sample_data from original data (EEG_data_PCA.txt)
     */
    FILE *data = fopen("EEG_data_PCA.txt", "r");

    char line[MAX_TIME];
    char *pos;
    int i = 0;
    if (data != NULL)
    {
        while (fgets(line, MAX_TIME, data))
        {
            int j = 0;
            char *tmp;
            pos = strtok(line, ",");
            while (pos != NULL)
            {
                sample_data[j][i] = strtod(pos, &tmp);
                j++;
                pos = strtok(NULL, ",");
            }
            i++;
        }
    }
    fclose(data);

    /*
     * Random initial weight matrix generator
     * Each values are -5 ~ 5
     */
    double weight[CH_NUM][CH_NUM];
    // srand((unsigned int)time(NULL));
    srand(1000);
    double sum = 0.0;
    for (int i = 0; i < CH_NUM; i++)
    {
        for (int j = 0; j < CH_NUM; j++)
        {
            int num = rand();
            num = num % 20001;
            weight[i][j] = (num - 10000.0);
            sum += pow(weight[i][j], 2);
        }
    }
    for (int i = 0; i < CH_NUM; i++)
    {
        for (int j = 0; j < CH_NUM; j++)
        {
            weight[i][j] /= sqrt(sum);
        }
    }

    // printf("Weight initial value\n");
    // for (int i = 0; i < CH_NUM; i++)
    // {
    //     for (int j = 0; j < CH_NUM; j++)
    //     {
    //         printf("%.2lf ", weight[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n\n");

    double weight_t[CH_NUM][CH_NUM];

    for (int learn_index = 0; learn_index < LEARN_COUNT; learn_index++)
    {
        /*
         * weight_t 만드는 부분
         */
        for (int r = 0; r < CH_NUM; r++)
        {
            for (int c = 0; c < CH_NUM; c++)
            {
                weight_t[c][r] = weight[r][c];
            }
        }

        double sum_mat[CH_NUM][CH_NUM] = {0};

        for (int t = 0; t < MAX_TIME; t++)
        {
            /*
             * x(t) 만드는 부분
             */
            double x[CH_NUM][1];
            for (int i = 0; i < CH_NUM; i++)
            {
                x[i][0] = sample_data[i][t];
            }

            /*
             * x^T (transpose) 만드는 부분
             */
            double xT[1][CH_NUM];
            for (int c = 0; c < CH_NUM; c++)
            {
                xT[0][c] = x[c][0];
            }

            /*
             * y(t) = Wx(t) 만들기
             */
            double y[CH_NUM][1];
            for (int i = 0; i < CH_NUM; i++)
            {
                for (int j = 0; j < CH_NUM; j++)
                {
                    y[i][0] += weight[i][j] * x[j][0];
                }
            }

            /*
             * 1 - 2 * sigmoid(y) 적용
             */
            for (int i = 0; i < CH_NUM; i++)
            {
                y[i][0] = 1.0 - 2.0 * sigmoid(y[i][0]);
            }

            /*
             * x(t)^T * W^T 계산
             */
            double xT_WT[1][CH_NUM] = {0};
            for (int j = 0; j < CH_NUM; j++)
            {
                for (int k = 0; k < CH_NUM; k++)
                {
                    xT_WT[0][j] += (xT[0][k] * weight_t[k][j]);
                }
            }

            /*
             * (y * xT_WT) 계산
             */
            for (int i = 0; i < CH_NUM; i++)
            {
                for (int j = 0; j < CH_NUM; j++)
                {
                    sum_mat[i][j] += (2.0 / MAX_TIME * (y[i][0] * xT_WT[0][j]));
                }
            }
        }

        /*
         * Sum + I
         */
        for (int i = 0; i < CH_NUM; i++)
        {
            for (int j = 0; j < CH_NUM; j++)
            {
                if (i == j)
                {
                    sum_mat[i][j] += 1;
                }
            }
        }

        /*
         * Gradient 계산
         */
        double gradient[CH_NUM][CH_NUM];
        for (int i = 0; i < CH_NUM; i++)
        {
            for (int j = 0; j < CH_NUM; j++)
            {
                for (int k = 0; k < CH_NUM; k++)
                {
                    gradient[i][j] += sum_mat[i][k] * weight[k][j];
                }
                gradient[i][j] *= ALPHA;
            }
        }

        /*
         * Weight 업데이트 & normalizing
         */
        double norm = 0.0;
        for (int i = 0; i < CH_NUM; i++)
        {
            for (int j = 0; j < CH_NUM; j++)
            {
                weight[i][j] += gradient[i][j];
                norm += pow(weight[i][j], 2);
            }
        }
        for (int i = 0; i < CH_NUM; i++)
        {
            for (int j = 0; j < CH_NUM; j++)
            {
                weight[i][j] /= sqrt(norm);
            }
        }
    }

    // printf("Weight learned value\n");
    // for (int i = 0; i < CH_NUM; i++)
    // {
    //     for (int j = 0; j < CH_NUM; j++)
    //     {
    //         printf("%.2lf ", weight[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n\n");

    /*
     * source data 추출
     */
    double extracted_source[CH_NUM][MAX_TIME];
    for (int i = 0; i < CH_NUM; i++)
    {
        for (int j = 0; j < MAX_TIME; j++)
        {
            for (int k = 0; k < CH_NUM; k++)
            {
                extracted_source[i][j] += (weight[i][k] * sample_data[k][j]);
            }
        }
    }

    for (int i = 0; i < MAX_TIME; i++)
    {
        for (int j = 0; j < CH_NUM; j++)
        {
            printf("%.4lf,", extracted_source[j][i]);
        }
        printf("\n");
    }

    // for (int t = 0; t < 1; t++)
    // {
    //     /*
    //      * weight_t 만드는 부분
    //      */
    //     for (int r = 0; r < CH_NUM; r++)
    //     {
    //         for (int c = 0; c < CH_NUM; c++)
    //         {
    //             weight_t[c][r] = weight[r][c];
    //         }
    //     }

    //     /*
    //      * x(t) 만드는 부분
    //      */
    //     double x[CH_NUM][1];
    //     for (int i = 0; i < CH_NUM; i++)
    //     {
    //         x[i][0] = sample_data[i][t];
    //     }

    //     /*
    //      * x^T (transpose) 만드는 부분
    //      */
    //     double xT[1][CH_NUM];
    //     for (int c = 0; c < CH_NUM; c++)
    //     {
    //         xT[0][c] = x[c][0];
    //     }

    //     double xT_WT[1][CH_NUM] = {0};
    //     for (int j = 0; j < CH_NUM; j++)
    //     {
    //         for (int k = 0; k < CH_NUM; k++)
    //         {
    //             xT_WT[0][j] += (xT[0][k] * weight_t[k][j]);
    //         }
    //     }

    //     //! 여기까지 이상 없음!

    //     double sigmoid_mat[CH_NUM][1];
    //     for (int i = 0; i < CH_NUM; i++)
    //     {
    //         double sum = 0.0;
    //         for (int j = 0; j < CH_NUM; j++)
    //         {
    //             sum += weight[i][j] * x[j][0];
    //         }
    //         sigmoid_mat[i][0] = 1.0 - (2.0 * sigmoid(sum));
    //     }

    //     printf("sigmoid matrix\n");
    //     for (int i = 0; i < CH_NUM; i++)
    //     {
    //         for (int j = 0; j < 1; j++)
    //         {
    //             printf("%.2lf ", sigmoid_mat[i][j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n\n");

    //     double temp_mat[CH_NUM][CH_NUM] = {0};
    //     for (int i = 0; i < CH_NUM; i++)
    //     {
    //         for (int j = 0; j < CH_NUM; j++)
    //         {
    //             temp_mat[i][j] = sigmoid_mat[i][0] * xT_WT[0][j];
    //         }
    //     }

    //     for (int i = 0; i < CH_NUM; i++)
    //     {
    //         for (int j = 0; j < CH_NUM; j++)
    //         {
    //             if (i == j)
    //             {
    //                 temp_mat[i][j] += 1;
    //             }
    //             temp_mat[i][j] *= alpha;
    //         }
    //     }

    //     double result_mat[CH_NUM][CH_NUM] = {0};
    //     for (int i = 0; i < CH_NUM; i++)
    //     {
    //         for (int j = 0; j < CH_NUM; j++)
    //         {
    //             for (int k = 0; k < CH_NUM; k++)
    //             {
    //                 result_mat[i][j] += temp_mat[i][k] * weight[k][j];
    //             }
    //         }
    //     }
    //     double sum_norm = 0.0;
    //     for (int i = 0; i < CH_NUM; i++)
    //     {
    //         for (int j = 0; j < CH_NUM; j++)
    //         {
    //             weight[i][j] += result_mat[i][j];
    //             sum_norm += pow(weight[i][j], 2);
    //         }
    //     }
    //     sum_norm = sqrt(sum_norm);
    //     for (int i = 0; i < CH_NUM; i++)
    //     {
    //         for (int j = 0; j < CH_NUM; j++)
    //         {
    //             weight[i][j] /= sum_norm;
    //         }
    //     }
    // }

    // for (int i = 0; i < CH_NUM; i++)
    // {
    //     for (int j = 0; j < CH_NUM; j++)
    //     {
    //         printf("%.2lf ", weight[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n\n");

    //* Source data 추출
    // for (int i = 0; i < CH_NUM; i++)
    // {
    //     for (int j = 0; j < 10000; j++)
    //     {
    //         for (int k = 0; k < CH_NUM; k++)
    //         {
    //             source_data[i][j] += (weight[i][k] * sample_data[k][j]);
    //         }
    //     }
    // }

    // for (int i = 0; i < CH_NUM; i++)
    // {
    //     for (int j = 0; j < CH_NUM; j++)
    //     {
    //         printf("%.2lf ", source_data[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n\n");

    return 0;
}
