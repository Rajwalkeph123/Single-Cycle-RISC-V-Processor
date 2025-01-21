`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 19.10.2024 19:30:38
// Design Name: 
// Module Name: q1
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
module SingleCycleCPU (
    input clk,      // Clock input for synchronous operations
    input start     // Input signal to start/reset the CPU
);

    // Define wires for connections between modules
    wire [31:0] pc_current, pc_next, pc_plus_4, instruction;
    wire [31:0] imm, branch_target;
    wire [31:0] read_data1, read_data2, write_data, alu_result, mem_data, alu_input2;
    wire [3:0] alu_control;
    wire zero, branch, memRead, memtoReg, memWrite, ALUSrc, regWrite;
    wire [1:0] ALUOp;
    
    // PC (Program Counter) Module
    PC m_PC(
        .clk(clk),
        .rst(~start),         // Reset the PC when start is low
        .pc_i(pc_next),       // Input the next PC value
        .pc_o(pc_current)     // Output the current PC value
    );

    // Adder for PC + 4
    Adder m_Adder_1(
        .a(pc_current),
        .b(32'd4),
        .sum(pc_plus_4)       // Output the next sequential PC value (PC + 4)
    );

    // Instruction Memory
    InstructionMemory m_InstMem(
        .readAddr(pc_current), // Read the instruction at the current PC address
        .inst(instruction)     // Output the fetched instruction
    );

    // Control Unit
    Control m_Control(
        .opcode(instruction[6:0]),
        .branch(branch),
        .memRead(memRead),
        .memtoReg(memtoReg),
        .ALUOp(ALUOp),
        .memWrite(memWrite),
        .ALUSrc(ALUSrc),
        .regWrite(regWrite)
    );

    // Register File
    Register m_Register(
        .clk(clk),
        .rst(~start),
        .regWrite(regWrite),
        .readReg1(instruction[19:15]),
        .readReg2(instruction[24:20]),
        .writeReg(instruction[11:7]),
        .writeData(write_data),
        .readData1(read_data1),
        .readData2(read_data2)
    );

    // Immediate Generator
    ImmGen #(.Width(32)) m_ImmGen(
        .inst(instruction),
        .imm(imm)  // Output the generated immediate value
    );

    // ALU Control
    ALUCtrl m_ALUCtrl(
        .ALUOp(ALUOp),
        .funct7(instruction[30]),
        .funct3(instruction[14:12]),
        .ALUCtl(alu_control)
    );

    // Mux for selecting ALU second operand (register or immediate)
    Mux2to1 #(.size(32)) m_Mux_ALU(
        .sel(ALUSrc),
        .s0(read_data2),  // Register value
        .s1(imm),         // Immediate value
        .out(alu_input2)  // Output to ALU input B
    );

    // ALU
    ALU m_ALU(
        .ALUCtl(alu_control),
        .A(read_data1),     // First ALU input
        .B(alu_input2),     // Second ALU input (either register value or immediate)
        .ALUOut(alu_result), // ALU result
        .zero(zero)         // Zero flag for branch decision
    );

    // Shift left the immediate value by 1 for branch calculation
    ShiftLeftOne m_ShiftLeftOne(
        .i(imm),
        .o(branch_target)  // Output the shifted value for branch address calculation
    );

    // Adder for calculating the branch target address
    Adder m_Adder_2(
        .a(pc_plus_4),
        .b(branch_target),
        .sum(pc_next)       // Output the target address for branching
    );

    // Mux for selecting the next PC value (branch or PC + 4)
    Mux2to1 #(.size(32)) m_Mux_PC(
        .sel(branch & zero), // Branch if the zero flag is set and the instruction is a branch
        .s0(pc_plus_4),      // Default to PC + 4
        .s1(pc_next),        // Use the branch target address if branching
        .out(pc_next)
    );

    // Data Memory
    DataMemory m_DataMemory(
        .rst(~start),
        .clk(clk),
        .memWrite(memWrite),
        .memRead(memRead),
        .address(alu_result),
        .writeData(read_data2),
        .readData(mem_data)
    );

    // Mux for selecting the data to write back to the register
    Mux2to1 #(.size(32)) m_Mux_WriteData(
        .sel(memtoReg),
        .s0(alu_result),  // Write ALU result
        .s1(mem_data),    // Write data read from memory
        .out(write_data)  // Output the data to write back to the register
    );

endmodule




module Adder (
    input signed [31:0] a,
    input signed [31:0] b,
    output signed [31:0] sum
);
    // Adder computes sum = a + b
    // The module is useful for incrementing PC 

 assign sum = a + b;

endmodule

module ALU (
    input [4:0] ALUCtl,       // 5-bit control signal to select the operation
    input [31:0] A, B,        // 32-bit inputs A and B for ALU operations
    output reg [31:0] ALUOut, // 32-bit output for the result of the ALU operation
    output zero              // Output flag to indicate if ALUOut is zero
);

    // Combinational logic to select the ALU operation based on ALUCtl
    always @(*) begin
        case (ALUCtl)
            5'b00000: ALUOut = A + B;          // ADD: Adds A and B
            5'b00001: ALUOut = A - B;          // SUB: Subtracts B from A
            5'b00010: ALUOut = A & B;          // AND: Bitwise AND of A and B
            5'b00011: ALUOut = A | B;          // OR: Bitwise OR of A and B
            5'b00100: ALUOut = A ^ B;          // XOR: Bitwise XOR of A and B
            5'b00101: ALUOut = A << B[4:0];    // SLL: Shift A left by the amount in B[4:0]
            5'b00110: ALUOut = A >> B[4:0];    // SRL: Shift A right logically by the amount in B[4:0]
            5'b00111: ALUOut = $signed(A) >>> B[4:0]; // SRA: Shift A right arithmetically by the amount in B[4:0]
            5'b01000: ALUOut = (A < B) ? 32'b1 : 32'b0;  // SLT: Set to 1 if A is less than B (signed comparison)
            5'b01001: ALUOut = ($unsigned(A) < $unsigned(B)) ? 32'b1 : 32'b0; // SLTU: Set to 1 if A is less than B (unsigned)
            default: ALUOut = 32'b0;           // Default: Output zero for undefined ALUCtl values
        endcase
    end

    // The zero flag is used to determine if ALUOut is zero
    // It will be 1 if ALUOut is 0, otherwise 0
    assign zero = (ALUOut == 32'b0) ? 1'b1 : 1'b0;

endmodule


module ALUCtrl (
    input [1:0] ALUOp,       // 2-bit input that determines the type of instruction
    input funct7,            // 1-bit input derived from the funct7 field of R-type instructions
    input [2:0] funct3,      // 3-bit input derived from the funct3 field
    output reg [3:0] ALUCtl  // 4-bit output that determines the specific ALU operation
);

    // Combinational logic to determine the ALU control signal based on ALUOp, funct7, and funct3
    always @(*) begin
        case (ALUOp)
            2'b00: ALUCtl = 4'b0010;  // LW, SW (I-type load/store): Use ALU for ADD (ALUOp = 00)
            2'b01: ALUCtl = 4'b0110;  // BEQ (branch): Use ALU for SUBTRACT (ALUOp = 01)
            2'b10: begin             // R-type instructions (ALUOp = 10)
                case (funct3)
                    3'b000: ALUCtl = (funct7 == 1'b0) ? 4'b0010 : 4'b0110; // ADD (funct7=0) or SUB (funct7=1)
                    3'b111: ALUCtl = 4'b0000;  // AND
                    3'b110: ALUCtl = 4'b0001;  // OR
                    3'b100: ALUCtl = 4'b0011;  // XOR
                    3'b001: ALUCtl = 4'b0101;  // SLL (Shift Left Logical)
                    3'b101: ALUCtl = (funct7 == 1'b0) ? 4'b0111 : 4'b1000; // SRL (funct7=0) or SRA (funct7=1)
                    3'b010: ALUCtl = 4'b1001;  // SLT (Set Less Than, signed)
                    3'b011: ALUCtl = 4'b1010;  // SLTU (Set Less Than, unsigned)
                    default: ALUCtl = 4'b0000; // Default to AND
                endcase
            end
            2'b11: begin             // I-type arithmetic instructions (ALUOp = 11)
                case (funct3)
                    3'b000: ALUCtl = 4'b0010;  // ADDI
                    3'b111: ALUCtl = 4'b0000;  // ANDI
                    3'b110: ALUCtl = 4'b0001;  // ORI
                    3'b100: ALUCtl = 4'b0011;  // XORI
                    3'b001: ALUCtl = 4'b0101;  // SLLI
                    3'b101: ALUCtl = (funct7 == 1'b0) ? 4'b0111 : 4'b1000; // SRLI or SRAI
                    3'b010: ALUCtl = 4'b1001;  // SLTI
                    3'b011: ALUCtl = 4'b1010;  // SLTIU
                    default: ALUCtl = 4'b0000; // Default to ANDI
                endcase
            end
            default: ALUCtl = 4'b0000;  // Default to AND if ALUOp doesn't match any case
        endcase
    end

endmodule


module Control (
    input [6:0] opcode,        // 7-bit opcode input to determine the instruction type
    output reg branch,         // Control signal for branch instructions
    output reg memRead,        // Control signal for reading from memory
    output reg memtoReg,       // Control signal for selecting between ALU result and memory read data
    output reg [1:0] ALUOp,    // 2-bit signal to determine the ALU operation type
    output reg memWrite,       // Control signal for writing to memory
    output reg ALUSrc,         // Control signal for choosing ALU input (immediate or register)
    output reg regWrite        // Control signal for writing to a register
);

    // Combinational logic to set control signals based on the opcode
    always @(*) begin
        // Default values for control signals (for unsupported instructions)
        branch = 0;
        memRead = 0;
        memtoReg = 0;
        memWrite = 0;
        ALUSrc = 0;
        regWrite = 0;
        ALUOp = 2'b00;

        case (opcode)
            7'b0110011: begin // R-type instructions (e.g., add, sub, and, or)
                ALUOp = 2'b10; // ALUOp = 10 for R-type
                regWrite = 1;  // Write the result to a register
                ALUSrc = 0;    // Use register data as the second ALU input
            end

            7'b0000011: begin // I-type load instruction (e.g., lw)
                memRead = 1;   // Read data from memory
                memtoReg = 1;  // Write memory data back to the register
                ALUOp = 2'b00; // ALUOp = 00 for addition (address calculation)
                regWrite = 1;  // Write the loaded value to a register
                ALUSrc = 1;    // Use immediate value as the second ALU input
            end

            7'b0100011: begin // S-type store instruction (e.g., sw)
                memWrite = 1;  // Write data to memory
                ALUOp = 2'b00; // ALUOp = 00 for addition (address calculation)
                ALUSrc = 1;    // Use immediate value as the second ALU input
            end

            7'b1100011: begin // B-type branch instruction (e.g., beq)
                branch = 1;    // Set the branch signal high
                ALUOp = 2'b01; // ALUOp = 01 for subtraction (used in branch condition)
            end

            7'b0010011: begin // I-type arithmetic instruction (e.g., addi)
                ALUOp = 2'b11; // ALUOp = 11 for I-type arithmetic instructions
                regWrite = 1;  // Write the result to a register
                ALUSrc = 1;    // Use immediate value as the second ALU input
            end

            default: begin
                // Keep all control signals as 0 for unsupported opcodes
                branch = 0;
                memRead = 0;
                memtoReg = 0;
                memWrite = 0;
                ALUSrc = 0;
                regWrite = 0;
                ALUOp = 2'b00;
            end
        endcase
    end

endmodule


module DataMemory(
	input rst,
	input clk,
	input memWrite,
	input memRead,
	input [31:0] address,
	input [31:0] writeData,
	output reg [31:0] readData
);
	// Do not modify this file!

	reg [7:0] data_memory [127:0];
	always @ (posedge clk) begin
		if(~rst) begin
			data_memory[0] <= 8'b0;
			data_memory[1] <= 8'b0;
			data_memory[2] <= 8'b0;
			data_memory[3] <= 8'b0;
			data_memory[4] <= 8'b0;
			data_memory[5] <= 8'b0;
			data_memory[6] <= 8'b0;
			data_memory[7] <= 8'b0;
			data_memory[8] <= 8'b0;
			data_memory[9] <= 8'b0;
			data_memory[10] <= 8'b0;
			data_memory[11] <= 8'b0;
			data_memory[12] <= 8'b0;
			data_memory[13] <= 8'b0;
			data_memory[14] <= 8'b0;
			data_memory[15] <= 8'b0;
			data_memory[16] <= 8'b0;
			data_memory[17] <= 8'b0;
			data_memory[18] <= 8'b0;
			data_memory[19] <= 8'b0;
			data_memory[20] <= 8'b0;
			data_memory[21] <= 8'b0;
			data_memory[22] <= 8'b0;
			data_memory[23] <= 8'b0;
			data_memory[24] <= 8'b0;
			data_memory[25] <= 8'b0;
			data_memory[26] <= 8'b0;
			data_memory[27] <= 8'b0;
			data_memory[28] <= 8'b0;
			data_memory[29] <= 8'b0;
			data_memory[30] <= 8'b0;
			data_memory[31] <= 8'b0;
			data_memory[32] <= 8'b0;
			data_memory[33] <= 8'b0;
			data_memory[34] <= 8'b0;
			data_memory[35] <= 8'b0;
			data_memory[36] <= 8'b0;
			data_memory[37] <= 8'b0;
			data_memory[38] <= 8'b0;
			data_memory[39] <= 8'b0;
			data_memory[40] <= 8'b0;
			data_memory[41] <= 8'b0;
			data_memory[42] <= 8'b0;
			data_memory[43] <= 8'b0;
			data_memory[44] <= 8'b0;
			data_memory[45] <= 8'b0;
			data_memory[46] <= 8'b0;
			data_memory[47] <= 8'b0;
			data_memory[48] <= 8'b0;
			data_memory[49] <= 8'b0;
			data_memory[50] <= 8'b0;
			data_memory[51] <= 8'b0;
			data_memory[52] <= 8'b0;
			data_memory[53] <= 8'b0;
			data_memory[54] <= 8'b0;
			data_memory[55] <= 8'b0;
			data_memory[56] <= 8'b0;
			data_memory[57] <= 8'b0;
			data_memory[58] <= 8'b0;
			data_memory[59] <= 8'b0;
			data_memory[60] <= 8'b0;
			data_memory[61] <= 8'b0;
			data_memory[62] <= 8'b0;
			data_memory[63] <= 8'b0;
			data_memory[64] <= 8'b0;
			data_memory[65] <= 8'b0;
			data_memory[66] <= 8'b0;
			data_memory[67] <= 8'b0;
			data_memory[68] <= 8'b0;
			data_memory[69] <= 8'b0;
			data_memory[70] <= 8'b0;
			data_memory[71] <= 8'b0;
			data_memory[72] <= 8'b0;
			data_memory[73] <= 8'b0;
			data_memory[74] <= 8'b0;
			data_memory[75] <= 8'b0;
			data_memory[76] <= 8'b0;
			data_memory[77] <= 8'b0;
			data_memory[78] <= 8'b0;
			data_memory[79] <= 8'b0;
			data_memory[80] <= 8'b0;
			data_memory[81] <= 8'b0;
			data_memory[82] <= 8'b0;
			data_memory[83] <= 8'b0;
			data_memory[84] <= 8'b0;
			data_memory[85] <= 8'b0;
			data_memory[86] <= 8'b0;
			data_memory[87] <= 8'b0;
			data_memory[88] <= 8'b0;
			data_memory[89] <= 8'b0;
			data_memory[90] <= 8'b0;
			data_memory[91] <= 8'b0;
			data_memory[92] <= 8'b0;
			data_memory[93] <= 8'b0;
			data_memory[94] <= 8'b0;
			data_memory[95] <= 8'b0;
			data_memory[96] <= 8'b0;
			data_memory[97] <= 8'b0;
			data_memory[98] <= 8'b0;
			data_memory[99] <= 8'b0;
			data_memory[100] <= 8'b0;
			data_memory[101] <= 8'b0;
			data_memory[102] <= 8'b0;
			data_memory[103] <= 8'b0;
			data_memory[104] <= 8'b0;
			data_memory[105] <= 8'b0;
			data_memory[106] <= 8'b0;
			data_memory[107] <= 8'b0;
			data_memory[108] <= 8'b0;
			data_memory[109] <= 8'b0;
			data_memory[110] <= 8'b0;
			data_memory[111] <= 8'b0;
			data_memory[112] <= 8'b0;
			data_memory[113] <= 8'b0;
			data_memory[114] <= 8'b0;
			data_memory[115] <= 8'b0;
			data_memory[116] <= 8'b0;
			data_memory[117] <= 8'b0;
			data_memory[118] <= 8'b0;
			data_memory[119] <= 8'b0;
			data_memory[120] <= 8'b0;
			data_memory[121] <= 8'b0;
			data_memory[122] <= 8'b0;
			data_memory[123] <= 8'b0;
			data_memory[124] <= 8'b0;
			data_memory[125] <= 8'b0;
			data_memory[126] <= 8'b0;
			data_memory[127] <= 8'b0;
		end
		else begin
			if(memWrite) begin
				data_memory[address + 3] <= writeData[31:24];
				data_memory[address + 2] <= writeData[23:16];
				data_memory[address + 1] <= writeData[15:8];
				data_memory[address]     <= writeData[7:0];
			end

			end
	end       

	always @(*) begin
		if(memRead) begin
			readData[31:24]   = data_memory[address + 3];
			readData[23:16]   = data_memory[address + 2];
			readData[15:8]    = data_memory[address + 1];
			readData[7:0]     = data_memory[address];
		end
		else begin
			readData          = 32'b0;
		end
	end

endmodule


module ImmGen#(parameter Width = 32) (
    input [Width-1:0] inst,               // 32-bit instruction input
    output reg signed [Width-1:0] imm     // 32-bit signed immediate output
);
    // Extract the opcode from the instruction
    wire [6:0] opcode = inst[6:0];

    always @(*) begin
        case(opcode)
            7'b0000011, 7'b0010011, 7'b1100111: begin
                // I-type instructions (e.g., addi, lw, jalr)
                imm = {{20{inst[31]}}, inst[31:20]}; // Sign-extend from bit [31:20]
            end

            7'b0100011: begin
                // S-type instructions (e.g., sw)
                imm = {{20{inst[31]}}, inst[31:25], inst[11:7]}; // Combine and sign-extend
            end

            7'b1100011: begin
                // B-type instructions (e.g., beq, bne)
                imm = {{19{inst[31]}}, inst[31], inst[7], inst[30:25], inst[11:8], 1'b0}; // Combine and sign-extend, with bit [0] as 0
            end

            7'b0110111, 7'b0010111: begin
                // U-type instructions (e.g., lui, auipc)
                imm = {inst[31:12], 12'b0}; // Shift left by 12 bits (upper immediate)
            end

            7'b1101111: begin
                // J-type instructions (e.g., jal)
                imm = {{11{inst[31]}}, inst[31], inst[19:12], inst[20], inst[30:21], 1'b0}; // Combine and sign-extend, with bit [0] as 0
            end

            default: begin
                imm = 0; // Default case for unsupported opcodes
            end
        endcase
    end
            
endmodule


module InstructionMemory (
    input [31:0] readAddr,
    output [31:0] inst
);
    
    // Do not modify this file!

    reg [7:0] insts [127:0];
    
    assign inst = (readAddr >= 128) ? 32'b0 : {insts[readAddr], insts[readAddr + 1], insts[readAddr + 2], insts[readAddr + 3]};

    initial begin
        insts[0] = 8'b0;  insts[1] = 8'b0;  insts[2] = 8'b0;  insts[3] = 8'b0;
        insts[4] = 8'b0;  insts[5] = 8'b0;  insts[6] = 8'b0;  insts[7] = 8'b0;
        insts[8] = 8'b0;  insts[9] = 8'b0;  insts[10] = 8'b0; insts[11] = 8'b0;
        insts[12] = 8'b0; insts[13] = 8'b0; insts[14] = 8'b0; insts[15] = 8'b0;
        insts[16] = 8'b0; insts[17] = 8'b0; insts[18] = 8'b0; insts[19] = 8'b0;
        insts[20] = 8'b0; insts[21] = 8'b0; insts[22] = 8'b0; insts[23] = 8'b0;
        insts[24] = 8'b0; insts[25] = 8'b0; insts[26] = 8'b0; insts[27] = 8'b0;
        insts[28] = 8'b0; insts[29] = 8'b0; insts[30] = 8'b0; insts[31] = 8'b0;
        $readmemb("TEST_INSTRUCTIONS.dat", insts);
    end 
  

endmodule

module Mux2to1 #(
    parameter size = 32
) 
(
    input sel,
    input signed [size-1:0] s0,
    input signed [size-1:0] s1,
    output signed [size-1:0] out
);
    
assign out = sel ? s1 : s0;
    
endmodule

module PC (
    input clk,
    input rst,
    input [31:0] pc_i,
    output reg [31:0] pc_o
);

always @(posedge clk ) begin
	if (~rst)
		pc_o <=32'b0;
	else
		pc_o <= pc_i;
end
endmodule

// Do not modify this file!

module Register (
    input clk,
    input rst,
    input regWrite,
    input [4:0] readReg1,
    input [4:0] readReg2,
    input [4:0] writeReg,
    input [31:0] writeData,
    output [31:0] readData1,
    output [31:0] readData2
);
    reg [31:0] regs [0:31];

// Do not modify this file!
    assign readData1 = (readReg1!=0)?regs[readReg1]:0;
    assign readData2 = (readReg2!=0)?regs[readReg2]:0;
     
    always @(posedge clk) begin
        if(~rst) begin
            regs[0] <= 0; regs[1] <= 0; regs[2] <= 32'd128; regs[3] <= 0; 
            regs[4] <= 0; regs[5] <= 0; regs[6] <= 0; regs[7] <= 0; 
            regs[8] <= 0; regs[9] <= 0; regs[10] <= 0; regs[11] <= 0; 
            regs[12] <= 0; regs[13] <= 0; regs[14] <= 0; regs[15] <= 0; 
            regs[16] <= 0; regs[17] <= 0; regs[18] <= 0; regs[19] <= 0; 
            regs[20] <= 0; regs[21] <= 0; regs[22] <= 0; regs[23] <= 0; 
            regs[24] <= 0; regs[25] <= 0; regs[26] <= 0; regs[27] <= 0; 
            regs[28] <= 0; regs[29] <= 0; regs[30] <= 0; regs[31] <= 0;        
        end
        else if(regWrite)
            regs[writeReg] <= (writeReg == 0) ? 0 : writeData;
    end

endmodule

module ShiftLeftOne (
    input signed [31:0] i,
    output signed [31:0] o
);

   assign o = i << 1;

endmodule




