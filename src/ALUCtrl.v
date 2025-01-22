module ALUCtrl (
    input [1:0] ALUOp,
    input funct7,
    input [2:0] funct3,
    output reg [3:0] ALUCtl
);

always @(*) begin
    case(ALUOp)
        2'b00: ALUCtl = 4'b0010; // ADD for LW/SW
        2'b01: ALUCtl = 4'b0110; // SUB for BEQ
        2'b10: begin
            case({funct7, funct3})
                4'b0000: ALUCtl = 4'b0010; // ADD
                4'b1000: ALUCtl = 4'b0110; // SUB
                4'b0111: ALUCtl = 4'b0000; // AND
                4'b0110: ALUCtl = 4'b0001; // OR
                4'b0010: ALUCtl = 4'b0111; // SLT
                default: ALUCtl = 4'b0000; // Default
            endcase
        end
        2'b11: begin  // Immediate-type (I-type) instructions (like ADDI, ORI, etc.)
            case(funct3)
                3'b000: ALUCtl = 4'b0010;  // ADDI (addition with immediate)
                3'b111: ALUCtl = 4'b0000;  // ANDI (AND with immediate)
                3'b110: ALUCtl = 4'b0001;  // ORI  (OR with immediate)
                3'b010: ALUCtl = 4'b0111;  // SLTI (Set Less Than Immediate)
                // Add other immediate-type operations (shift, etc.) as needed
                default: ALUCtl = 4'b0000; // Default case for unknown immediate instructions
            endcase
        end
        default: ALUCtl = 4'b0000; // Default case
    endcase
end


endmodule

