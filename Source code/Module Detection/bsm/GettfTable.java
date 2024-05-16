package bsm;

import java.util.HashSet;
import java.util.Set;

public class GettfTable extends Thread {
	private int startNum;
	RegulatorBindingData bindingData;
	int numrows;
	int segment;
	boolean[] isSeed;
	public static double[][] coregulatevalue;
	public static int[][] coregulateindex;
	public GettfTable(RegulatorBindingData bindingData, boolean[] isSeed, int numrows, int segment, int startNum) {
		this.startNum = startNum;
		this.bindingData = bindingData;
		this.segment = segment;
		this.numrows = numrows;
		this.isSeed = isSeed;
	}
	
	public void run() {
		if(bindingData != null){
			double[][] subvalue = new double[Math.min(segment, numrows-startNum)][];
			int[][] subindex = new int[Math.min(segment, numrows-startNum)][];
			makeCoregulatorIndex(bindingData.gene2RegBindingIndex, bindingData.gene2RegBinding, 
					bindingData.reg2GeneBindingIndex, bindingData.reg2GeneBinding, 
					subindex, subvalue, startNum, Math.min(startNum+segment, numrows));
			//System.out.println(subvalue[0].length);
			for(int i=startNum;i<Math.min(startNum+segment, numrows);i++){
				coregulatevalue[i] = subvalue[i-startNum];
				coregulateindex[i] = subindex[i-startNum];
			}
		}
	}
	
	public void multithread(RegulatorBindingData bindingData, boolean[] isSeed, int numrows, int segment) {
		int divide = numrows/segment;
		coregulatevalue = new double[numrows][];
		coregulateindex = new int[numrows][];
		
		Thread[] threadList = new Thread[divide+1];
		for (int i = 0; i < divide+1; i++) {
			threadList[i] = new GettfTable(bindingData, isSeed, numrows, segment, segment * i);
			threadList[i].start();
		}
		for (int i = 0; i < divide+1; i++) {
			try {
				threadList[i].join();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	private void makeCoregulatorIndex(int[][] valIndex, double[][] val,
			int[][] RegIndex, double[][] Reg, int[][] coregulateindex,
			double[][] coregulatevalue, int start, int end) {
		for(int i=start;i<end;i++){
			if(!isSeed[i] && valIndex[i]!=null && valIndex[i].length>0){
				for(int j=0;j<valIndex[i].length;j++){
					int nReg = valIndex[i][j];
					double nval = val[i][j];
					int[] list1 = RegIndex[nReg];
					double[] value1 = Reg[nReg];
					if(list1.length > 1){
						if(coregulateindex[i-start] != null){
							regulationset re = coupdate(i, nval, list1, value1, 
									coregulateindex[i-start], coregulatevalue[i-start]);
							coregulateindex[i-start] = re.geneindex;
							coregulatevalue[i-start] = re.value;	
						}else{
							int[] index = new int[list1.length];
							double[] value = new double[list1.length];
							int count=0;
							for(int p=0;p<list1.length;p++){
								if(list1[p] != i && isSeed[list1[p]]){
									index[count] = list1[p];
									value[count] = Math.abs(nval*value1[p]);
									count++;
								}
						   }
							coregulateindex[i-start] = new int[count];
							coregulatevalue[i-start] = new double[count];
							for(int p=0;p<count;p++){
								coregulateindex[i-start][p] = index[p];
								coregulatevalue[i-start][p] = value[p]/count;
							}
					   }
					}
				}
				if(coregulatevalue[i-start] != null){
					double sum = 0;
					for(int j=0;j<coregulatevalue[i-start].length;j++){
						sum += coregulatevalue[i-start][j];
					}
					for(int j=0;j<coregulatevalue[i-start].length;j++){
						coregulatevalue[i-start][j] /= sum;
					}
				}
			}
			}
		}
	static class regulationset{
		int[] geneindex;
		double[] value;
		regulationset(int[] geneindex, double[] value){
			this.geneindex = geneindex;
			this.value = value;
		}
	}
	private regulationset coupdate(int geneindex, double nval, int[] list1, double[] value1, 
			int[] coindex, double[] covalue){
		Set<Integer> set = new HashSet<Integer>();
		for(int i:coindex) set.add(i);
		int sum = 0;
		for(int i:list1){
			if(i != geneindex && isSeed[i]){
				sum++;
				set.add(i);
			}
		}
		int[] list = new int[set.size()];
		double[] value = new double[set.size()];
		int count = 0;
		for(int gene:set){
			list[count] = gene;
			for(int i=0;i<coindex.length;i++){
				if(coindex[i] == gene){
					value[count] = covalue[i];
					break;
				}
			}
			for(int i=0;i<list1.length;i++){
				if(list1[i] == gene){
					value[count] += Math.abs(nval*value1[i]) / sum;
					break;
				}
			}
			count++;
		}
		regulationset re = new regulationset(list, value);
		return re;
    }
}