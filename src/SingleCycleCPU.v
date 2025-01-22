module SingleCycleCPU (
    input clk,            // Clock signal
    input start           // Start signal (reset control)
);
    wire [31:0] pc, next_pc, instruction, readData1, readData2, aluResult, imm, writeData, aluIn2, readDataMem;
    wire [3:0] aluctl;
    wire aluSrc, branch, memRead, memtoReg, memWrite, regWrite, zero;
    wire [1:0] ALUOp;

    // Program Counter (PC)
    PC pcModule (
        .clk(clk),
        .rst(~start),
        .pc_i(next_pc),   // Next PC address (either PC+4 or branch address)
        .pc_o(pc)        // Current PC address
    );

    // Instruction Memory to fetch the instruction
    InstructionMemory instrMem (
        .readAddr(pc),
        .inst(instruction)   // Fetched instruction
    );

    // Control Unit that generates control signals based on the opcode
    Control controlUnit (
        .opcode(instruction[6:0]),  // Opcode from instruction
        .branch(branch),
        .memRead(memRead),
        .memtoReg(memtoReg),
        .ALUOp(ALUOp),
        .memWrite(memWrite),
        .ALUSrc(aluSrc),
        .regWrite(regWrite)
    );

    // Register File for reading/writing registers
    Register regFile (
        .clk(clk),
        .rst(~start),
        .regWrite(regWrite),
        .readReg1(instruction[19:15]),  // rs1 field
        .readReg2(instruction[24:20]),  // rs2 field
        .writeReg(instruction[11:7]),   // rd field
        .writeData(writeData),          // Data to write to rd
        .readData1(readData1),          // Data from rs1
        .readData2(readData2)           // Data from rs2
    );

    // Immediate Generator
    ImmGen immGen (
        .inst(instruction),  // Instruction input
        .imm(imm)            // Immediate output
    );

    // ALU Control for determining the ALU operation
    ALUCtrl aluControl (
        .ALUOp(ALUOp),
        .funct7(instruction[30]),      // funct7 from instruction
        .funct3(instruction[14:12]),   // funct3 from instruction
        .ALUCtl(aluctl)                // ALU control output
    );

    // Mux to select between register and immediate value for ALU operand B
    Mux2to1 aluSrcMux (
        .sel(aluSrc),
        .s0(readData2),  // Register value (rs2)
        .s1(imm),        // Immediate value
        .out(aluIn2)     // ALU input B
    );

    // ALU execution
    ALU alu (
        .ALUCtl(aluctl),  // ALU control signal
        .A(readData1),    // ALU input A (rs1)
        .B(aluIn2),       // ALU input B (either rs2 or immediate)
        .ALUOut(aluResult), // ALU result
        .zero(zero)       // Zero flag for branch decision
    );

    // Data Memory for memory access (load/store)
    DataMemory dataMemory (
        .rst(~start),
        .clk(clk),
        .memWrite(memWrite),     // Memory write enable
        .memRead(memRead),       // Memory read enable
        .address(aluResult),     // Memory address (ALU result)
        .writeData(readData2),   // Data to write (rs2)
        .readData(readDataMem)   // Data read from memory
    );

    // Mux to select between ALU result and data from memory for register write-back
    Mux2to1 wbMux (
        .sel(memtoReg),
        .s0(aluResult),    // ALU result
        .s1(readDataMem),  // Data read from memory
        .out(writeData)    // Data to write back to the register
    );

    // PC calculation: next PC is either PC+4 or branch target
    wire [31:0] pcPlus4, branchAddr;
    Adder pcAdder (
        .a(pc),
        .b(32'd4),         // Increment PC by 4
        .sum(pcPlus4)      // PC + 4
    );

    // Shift left by 1 for branch address calculation

    // Branch address calculation
    Adder branchAdder (
        .a(pc),
        .b(imm),
        .sum(branchAddr)   // Branch target address
    );

    // Mux to select next PC: either PC+4 or branch address
    Mux2to1 branchMux (
        .sel(branch & zero),  // Take the branch if branch is true and zero flag is set
        .s0(pcPlus4),         // PC + 4
        .s1(branchAddr),      // Branch target address
        .out(next_pc)         // Next PC address
    );

endmodule
