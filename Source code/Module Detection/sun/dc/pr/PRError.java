package sun.dc.pr;

 public class PRError extends java.lang.RuntimeException
{

    public static final String
         UNEX_setUsage = "setUsage: unexpected",
         UNEX_setFillMode = "setFillMode: unexpected",
         UNEX_setPenDiameter = "setPenDiameter: unexpected",
         UNEX_setPenT4 = "setPenT4: unexpected",
         UNEX_setPenDisplacement = "setPenDisplacement: unexpected",
         UNEX_setPenFitting = "setPenFitting: unexpected",
         UNEX_setCaps = "setCaps: unexpected",
         UNEX_setCorners = "setCorners: unexpected",
         UNEX_setDash = "setDash: unexpected",
         UNEX_setDashT4 = "setDashT4: unexpected",
         UNEX_beginPath = "beginPath: unexpected",
         UNEX_beginSubpath = "beginSubpath: unexpected",
         UNEX_appendCubic = "appendCubic: unexpected",
         UNEX_appendLine = "appendLine: unexpected",
         UNEX_appendQuadratic = "appendQuadratic: unexpected",
         UNEX_closedSubpath = "closedSubpath: unexpected",
         UNEX_endPath = "endPath: unexpected",
         UNEX_useProxy = "useProxy: unexpected",
         UNEX_setOutputConsumer = "setOutputConsumer: unexpected",
         UNEX_setOutputT6 = "setOutputT6: unexpected",
         UNEX_getAlphaBox = "getAlphaBox: unexpected",
         UNEX_setOutputArea = "setOutputArea: unexpected",
         UNEX_getTileState = "getTileState: unexpected",
         UNEX_writeAlpha = "writeAlpha: unexpected",
         UNEX_nextTile = "nextTile: unexpected",
         UNK_usage = "setUsage: unknown usage type",
         UNK_fillmode = "setFillMode: unknown fill mode",
         BAD_pendiam = "setPenDiameter: Invalid pen diameter",
         BAD_pent4 = "setPenT4: invalid pen transformation",
         BAD_pent4_singular = "setPenT4: invalid pen transformation (singular)",
         BAD_penfit = "setPenFitting: invalid pen fitting specification",
         UNK_caps = "setCaps: unknown cap type",
         UNK_corners = "setCorners: unknown corner type",
         BAD_miterlimit = "setCorners: invalid miter limit",
         BAD_dashpattern = "setDash: invalid dash pattern",
         BAD_dasht4 = "setDashT4: invalid dash transformation",
         BAD_dasht4_singular = "setDashT4: invalid dash transformation (singular)",
         BAD_pathbox = "beginPath: invalid path box",
         BAD_outputt6 = "setOutputT6: invalid output transformation",
         BAD_outputt6_singular = "setOutputT6: invalid output transformation (singular)",
         BAD_boxdest = "getAlphaBox: invalid box destination array",
         BAD_outputarea = "setOutputArea: invalid output area",
         BAD_alphadest = "writeAlpha: invalid alpha destination array and/or strides",
         DUMMY = "";

    // Constructors
     public PRError() {
        super();
    }

     public PRError(String s) {
        super(s);
    }
}

