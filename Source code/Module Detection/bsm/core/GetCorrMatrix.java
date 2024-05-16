package bsm.core;

public class GetCorrMatrix extends Thread {
	private int startNum;
	double[][] data;
	int numrows;
	int segment;
	public static double Sumcorr, Sumcorrsq, Suncount;
	public GetCorrMatrix(double[][] data, int numrows, int segment, int startNum) {
		this.startNum = startNum;
		this.data = data;
		this.segment = segment;
		this.numrows = numrows;
	}
	public GetCorrMatrix() {
		// TODO Auto-generated constructor stub
	}
	public synchronized void add(double sumcorr, double sumcorrsq, double suncount) {
		//System.out.println(sub.length);
		Sumcorr += sumcorr;
		Sumcorrsq += sumcorrsq;
		Suncount += suncount;
	}
	public void run() {
		Pearsonr pr = new Pearsonr();
		double sumcorr = 0, sumcorrsq = 0, suncount = 0;
		double[] exp1, exp2;
		for (int i = startNum; i < Math.min(startNum+segment, numrows); i++){
			exp1 = data[i];
			for(int j=i+1;j<numrows;j++){
				exp2 = data[j];
				double corr = pr.cosineSimilarity(exp1, exp2);			
				sumcorr += corr;
				sumcorrsq += Math.pow(corr, 2);
				suncount++;
			}
		}
		add(sumcorr, sumcorrsq, suncount);
	}
	public void multithread(double[][] data, int numrows, int segment) {
		int divide = numrows/segment;
		Sumcorr = 0;   
		Sumcorrsq = 0;
		
		Thread[] threadList = new Thread[divide+1];
		for (int i = 0; i < divide+1; i++) {
			threadList[i] = new GetCorrMatrix(data, numrows, segment, segment * i);
			/* ���ｫ�ֱ���ÿ��ĳ�ʼ���� */
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
	/*
	public static void main(String[] args) {
		double[][] a = null;
		multithread(a);
	}
	*/
}
