#define _CRT_NONSTDC_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <fstream>
#include <cmath>
#include <vector>
using namespace std;

int to8(int a) {
	string res = "";
	while (a > 0) {
		res = to_string(a % 8) + res;
		a /= 8;
	}
	return stoi(res);
}

void GenerateDataset(ostream &fout, const int num) {
	srand(time(0));
	for (int i = 0; i < num; i++) {
		int temp;
		while(true) {
			temp = rand() % 512;
			if(temp > 63)
				break;
		}
		fout << to8(temp) << "\n";
	}
}

int SortDataset(vector <int> &a) {
	int count = 0;
	int num = a.size();
	int d = num / 2;
	while (d > 0) {
		for (int i = 0; i < num - d; i++) {
			int j = i;
			while (j >= 0 && a[j] > a[j + d]) {
				int temp = a[j];
				a[j] = a[j + d];
				a[j + d] = temp;
				count++;
				j--;
			}
			//count++;
		}
		//count++;
		d = d / 2;
	}
	return count;
}
int main() {
	/*
	const char* file = "input.txt";
	const char* file_sort = "output.txt";
	const char* file_compares = "table.txt";
	*/
	ofstream fout, fout_table;
	ifstream fin;
	fout_table.open("table.txt");
	for (int num = 8; num <= 4096; num *= 2) {

		fout.open("input.txt");
		GenerateDataset(fout, num);
		fout.close();

		fin.open("input.txt");
		vector <int> a(num);
		for(int i = 0; i < num; ++i) {
			fin >> a[i];
		}
		fin.close();

		int count = SortDataset(a);

		fout.open("output.txt");
		for(auto &x : a)
			fout << x << "\n";
		fout.close();

		fout_table << num << "\t\t\t" << count << "\t\t\t";
		fout_table << num * num << "\t\t\t" << num * log2(num) << "\t\t\t";
		fout_table << (count + 0.0) / (num * num) << "\t\t\t";
		fout_table << (count + 0.0) / (num * log2(num)) << "\n";
		
		fout.open("input.txt");
		fout.close();
		fout.open("output.txt");
		fout.close();
	}
	fout_table.close();
	return 0;
}
