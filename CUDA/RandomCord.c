#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float RandomNumber(float Min, float Max)
{
	return (((float)(rand()) / (float)(RAND_MAX)) * (Max)) + Min;
}

int main()
{
	FILE * fptr;
	int i, n;;
	char str1;
	//for range from -5 to 5
	float Min = -5;
	float Max = 10;
	printf("\n\n Write multiple coordinates in a text file and read the file :\n");
	printf("------------------------------------------------------------\n");
	printf(" Input the number of lines to be written : ");
	scanf("%d", &n);
	//https://stackoverflow.com/questions/29154056/redirect-stdout-to-a-file
	freopen("test.txt", "a+", stdout);
	//fptr = fopen(fname, "w");
	for (i = 1; i < n+1; i++)
	{
		printf("%i", i);
		for (int j = 0; j < 3; j++)
		{
			printf(" %f", RandomNumber(Min,Max));
		}
		printf("\n");
	}
	freopen("CON", "w", stdout);

	/*-------------- read the file -------------------------------------*/
	fptr = fopen("test.txt", "r");
	printf("\n The content of the file %s is  :\n", "test.txt");
	str1 = fgetc(fptr);
	while (str1 != EOF)
	{
		printf("%c", str1);
		str1 = fgetc(fptr);
	}
	printf("\n\n");
	fclose(fptr);
	system("pause");
	return 0;
}
