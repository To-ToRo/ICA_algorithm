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
#define ALPHA 1.5

double sigmoid(double x)
{
    return (1 / (1 + exp(-x)));
}

double sample_data[CH_NUM][10000];
double source_data[CH_NUM][10000];

int main(void)
{
    /*
     * Generate sample_data from original data (EEG_data_PCA.txt)
     */
    FILE *data = fopen("EEG_data_PCA.txt", "r");

    char line[10000];
    char *pos;
    int i = 0;
    if (data != NULL)
    {
        while (fgets(line, 10000, data))
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
    srand((unsigned int)time(NULL));
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
    sum = sqrt(sum);
    for (int i = 0; i < CH_NUM; i++)
    {
        for (int j = 0; j < CH_NUM; j++)
        {
            weight[i][j] /= sum;
        }
    }

    double alpha = 1;

    for (int i = 0; i < CH_NUM; i++)
    {
        for (int j = 0; j < CH_NUM; j++)
        {
            printf("%.2lf ", weight[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");

    double weight_t[CH_NUM][CH_NUM];
    for (int t = 0; t < 10000; t++)
    {
        for (int r = 0; r < CH_NUM; r++)
        {
            for (int c = 0; c < CH_NUM; c++)
            {
                weight_t[c][r] = weight[r][c];
            }
        }

        double x[CH_NUM][1];
        for (int i = 0; i < CH_NUM; i++)
        {
            x[i][0] = sample_data[i][t];
        }

        double xT[1][CH_NUM];
        for (int c = 0; c < CH_NUM; c++)
        {
            xT[0][c] = x[c][0];
        }

        double xT_WT[1][CH_NUM] = {0};
        for (int j = 0; j < CH_NUM; j++)
        {
            for (int k = 0; k < CH_NUM; k++)
            {
                xT_WT[0][j] += (xT[0][k] * weight_t[k][j]);
            }
        }

        double sigmoid_mat[CH_NUM][1];
        for (int i = 0; i < CH_NUM; i++)
        {
            double sum = 0;
            for (int j = 0; j < CH_NUM; j++)
            {
                sum += weight[i][j] * x[j][0];
            }

            sigmoid_mat[i][0] = 1.0 - (2.0 * sigmoid(sum));
        }

        double temp_mat[CH_NUM][CH_NUM] = {0};
        for (int i = 0; i < CH_NUM; i++)
        {
            for (int j = 0; j < CH_NUM; j++)
            {
                for (int k = 0; k < 1; k++)
                {
                    temp_mat[i][j] += sigmoid_mat[i][k] * xT_WT[k][j];
                }
            }
        }

        for (int i = 0; i < CH_NUM; i++)
        {
            for (int j = 0; j < CH_NUM; j++)
            {
                if (i == j)
                {
                    temp_mat[i][j] += 1;
                }
                temp_mat[i][j] *= alpha;
            }
        }

        double result_mat[CH_NUM][CH_NUM] = {0};
        for (int i = 0; i < CH_NUM; i++)
        {
            for (int j = 0; j < CH_NUM; j++)
            {
                for (int k = 0; k < CH_NUM; k++)
                {
                    result_mat[i][j] += temp_mat[i][k] * weight[k][j];
                }
            }
        }
        double sum_norm = 0.0;
        for (int i = 0; i < CH_NUM; i++)
        {
            for (int j = 0; j < CH_NUM; j++)
            {
                weight[i][j] += result_mat[i][j];
                sum_norm += pow(weight[i][j], 2);
            }
        }
        sum_norm = sqrt(sum_norm);
        for (int i = 0; i < CH_NUM; i++)
        {
            for (int j = 0; j < CH_NUM; j++)
            {
                weight[i][j] /= sum_norm;
            }
        }
    }

    for (int i = 0; i < CH_NUM; i++)
    {
        for (int j = 0; j < CH_NUM; j++)
        {
            printf("%.2lf ", weight[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");

    for (int i = 0; i < CH_NUM; i++)
    {
        for (int j = 0; j < 10000; j++)
        {
            for (int k = 0; k < CH_NUM; k++)
            {
                source_data[i][j] += (weight[i][k] * sample_data[k][j]);
            }
        }
    }

    for (int i = 0; i < CH_NUM; i++)
    {
        for (int j = 0; j < CH_NUM; j++)
        {
            printf("%.2lf ", source_data[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");

    return 0;
}
