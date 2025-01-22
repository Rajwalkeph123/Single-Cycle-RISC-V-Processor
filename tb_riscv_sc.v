module tb_riscv_sc;
//cpu testbench

reg clk;
reg start;

SingleCycleCPU uut(clk, start);

wire [31:0] i;
assign i = uut.instruction;

wire [31:0] s0, sp, t0, t1;

    // Accessing the registers directly from the SingleCycleCPU
    assign s0 = uut.regFile.regs[8];  // s0 is register x8
    assign sp = uut.regFile.regs[2];  // sp is register x2
    assign t0 = uut.regFile.regs[5];  // t0 is register x5
    assign t1 = uut.regFile.regs[6];  // t1 is register x6


initial
	forever #5 clk = ~clk;

initial begin
	clk = 0;
	start = 0;
	#10 start = 1;

	#450 $finish;

end

endmodule
