#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * The number of channel and the value of alpha
 * will be determined at hardware level design with Verilog.
 */
#define CH_NUM 16
#define ALPHA 1.5

int main(void)
{
    FILE *data = fopen("EEG_data_PCA.txt", "r");

    double sample_data[CH_NUM][10001];

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
    // FILE *temp_txt = fopen("EEG_data_PCA.csv", "r");

    for (int i = 0; i < CH_NUM; i++)
    {
        for (int j = 0; j < 10000; j++)
        {
            printf("%lf,", sample_data[i][j]);
        }
        printf("\n");
    }

    double weight[CH_NUM][CH_NUM] = {0};
    double alpha = 1.5;

    for (int i = 0; i < CH_NUM; i++)
    {
    }

    return 0;
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

void transpose(double **A, double **B, int row, int col)
{
    for (int r = 0; r < row; r++)
    {
        for (int c = 0; c < col; c++)
        {
            B[c][r] = A[r][c];
        }
    }
}

void matrix_multiply(double **A, double **B, double **C, int row1, int col1, int col2)
{
    //* Initializing
    for (int i = 0; i < row1; ++i)
    {
        for (int j = 0; j < col2; ++j)
        {
            C[i][j] = 0;
        }
    }

    for (int i = 0; i < row1; i++)
    {
        for (int j = 0; j < col2; j++)
        {
            for (int k = 0; k < col1; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
