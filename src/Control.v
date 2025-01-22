module Control (
    input [6:0] opcode,    // Opcode from the instruction
    output reg branch,      // Branch control signal
    output reg memRead,     // Memory read signal
    output reg memtoReg,    // Write data source (ALU result or memory)
    output reg [1:0] ALUOp, // ALU operation code
    output reg memWrite,    // Memory write signal
    output reg ALUSrc,      // ALU source (immediate or register)
    output reg regWrite      // Register write signal
);

always @(*) begin
    case(opcode)
        7'b0000011: begin // Load (LW)
            memRead = 1; memWrite = 0; regWrite = 1; ALUSrc = 1;
            branch = 0; memtoReg = 1; ALUOp = 2'b00; 
        end
        7'b0100011: begin // Store (SW)
            memRead = 0; memWrite = 1; regWrite = 0; ALUSrc = 1;
            branch = 0; memtoReg = 0; ALUOp = 2'b00; 
        end
        7'b1100011: begin // Branch (BEQ)
            memRead = 0; memWrite = 0; regWrite = 0; ALUSrc = 0;
            branch = 1; memtoReg = 0; ALUOp = 2'b01; 
        end
        7'b0010011: begin // I-type ALU instructions (e.g., ADDI)
            memRead = 0; memWrite = 0; regWrite = 1; ALUSrc = 1;
            branch = 0; memtoReg = 0; ALUOp = 2'b11; 
        end
        7'b0110011: begin // R-type
            memRead = 0; memWrite = 0; regWrite = 1; ALUSrc = 0;
            branch = 0; memtoReg = 0; ALUOp = 2'b10; 
        end
        default: begin
            memRead = 0; memWrite = 0; regWrite = 0; ALUSrc = 0;
            branch = 0; memtoReg = 0; ALUOp = 2'b00; 
        end
    endcase
end
endmodule
