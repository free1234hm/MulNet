package bsm;

import bsm.core.*;

import java.io.*;

/**
 * Class extends DataSetCore which contains the underlying data and parameters
 * of the methods with an instance of DREM_GoAnnotations
 */
public class BSM_DataSet extends DataSetCore {

	
	public BSM_DataSet(DataSetCore theDataSetCore) {
		super(theDataSetCore);
	}
	/**
	 * Constructor that takes input parameters and calls dataSetReader to read
	 * in the content of szInputFile
	 */
	public BSM_DataSet(String szInputFile, Integer nmaxmissing, Integer missinter, 
			boolean btakelog)throws IOException, FileNotFoundException, IllegalArgumentException {
		
		this.szInputFile = szInputFile;
		this.nmaxmissing = nmaxmissing;
		this.missinter = missinter;
		this.btakelog = btakelog;
		
		dataSetReader(szInputFile, missinter, btakelog);
		
	}
}
