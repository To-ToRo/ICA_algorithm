#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * The number of channel and the value of alpha
 * will be determined at hardware level design with Verilog.
 */
#define CH_NUM 16
#define ALPHA 1.5

int main(void)
{
    FILE *data = fopen("EEG_data_PCA.csv", "r");

    float sample_data[10001][CH_NUM];

    char line[10001];
    char *pos;
    int i = 0;
    while (fgets(line, 10000, data))
    {
        int j = 0;
        char *tmp;
        pos = strtok(line, ",");
        while (pos != NULL)
        {
            sample_data[i][j] = strtof(pos, &tmp);
            j++;
            pos = strtok(NULL, ",");
        }
        i++;
    }

        return 0;
}
