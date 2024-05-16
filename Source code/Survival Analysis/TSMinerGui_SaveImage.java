package bsm;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.imageio.*;
import java.awt.image.*;

/**
 * Class to encapsulate window used to specify a file to save a TSMiner model
 */ 
public class TSMinerGui_SaveImage extends JPanel
{
	/**
	 * Class constructor
	 */
	public TSMinerGui_SaveImage(final JFrame theFrame, JScrollPane panel)
	{
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		String[] sznames = ImageIO.getWriterFormatNames();
		JFileChooser theChooser = new JFileChooser();
		add(theChooser);
		theChooser.setDialogType(JFileChooser.SAVE_DIALOG);
		//theChooser.addChoosableFileFilter(new JAVAFileFilter("jpg"));
		//theChooser.addChoosableFileFilter(new JAVAFileFilter("png"));
		theChooser.addActionListener(new ActionListener() 
		{
			public void actionPerformed(ActionEvent e) 
			{
				// set label's icon to the current image
				String state = (String)e.getActionCommand();

				if (state.equals(JFileChooser.CANCEL_SELECTION))
				{
					theFrame.setVisible(false);
					theFrame.dispose();
				}
				else if (state.equals(JFileChooser.APPROVE_SELECTION))
				{
					File file = theChooser.getSelectedFile();
					try
					{
						String szext = DREMGui_ImageFilter.getExtension(file);
						
						if(szext==null || szext.length()==0){	
							File newFile = new File(file.getAbsolutePath() + ".png");
							Dimension imageSize = panel.getSize();
						    BufferedImage image = new BufferedImage(imageSize.width,imageSize.height, BufferedImage.TYPE_INT_RGB);
						    Graphics2D graphics = image.createGraphics();
						    panel.paint(graphics);
						    graphics.dispose();
							ImageIO.write(image, "png", newFile);
						}else{
							for(int i=0;i<sznames.length;i++){
								if(szext.equalsIgnoreCase(sznames[i])){
									File newFile = new File(file.getAbsolutePath() + szext);
									Dimension imageSize = panel.getSize();
								    BufferedImage image = new BufferedImage(imageSize.width,imageSize.height, BufferedImage.TYPE_INT_RGB);
								    Graphics2D graphics = image.createGraphics();
								    panel.paint(graphics);
								    graphics.dispose();
									ImageIO.write(image, szext, newFile);
									break;
								}
							}
						}
					}catch (final IOException fex){
						javax.swing.SwingUtilities.invokeLater(new Runnable() 
						{
							public void run() 
							{
								JOptionPane.showMessageDialog(null, fex.getMessage(), 
										"Exception thrown", JOptionPane.ERROR_MESSAGE);
							}
						});
						fex.printStackTrace(System.out);
					}
					theFrame.setVisible(false);
					theFrame.dispose();
				}
			}
		});			      
	}
	/*
	class JAVAFileFilter extends FileFilter {
	    String ext;

	    public JAVAFileFilter(String ext) {
	        this.ext = ext;
	    }

	   
	    public boolean accept(File file) {
	        if (file.isDirectory()) {
	            return true;
	        }
	        String fileName = file.getName();
	        int index = fileName.lastIndexOf('.');
	        if (index > 0 && index < fileName.length() - 1) {
	            // ��ʾ�ļ����Ʋ�Ϊ".xxx"��"xxx."֮����
	            String extension = fileName.substring(index + 1).toLowerCase();
	            // ����ץ�����ļ���չ����������������Ҫ��ʾ����չ��(������extֵ),�򷵻�true,��ʾ�����ļ���ʾ����,���򷵻�
	            // true.
	            if (extension.equals(ext))
	                return true;
	        }
	        return false;
	    }

	    // ʵ��getDescription()����,���������ļ���˵���ַ���!!!
	    public String getDescription() {
	        if (ext.equals("png")) return "ͼƬ(*.png)";
	        if (ext.equals("jpg")) return "ͼƬ(*.jpg)";
	        return "";
	    }
	}
	*/
}
