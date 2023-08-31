import React from "react";
import {
  Button,
  Dialog,
  DialogHeader,
  DialogBody,
  DialogFooter,
  Input,
  Textarea,
} from "@material-tailwind/react";
 
export function MessageDialog(args) {
  const [open, setOpen] = React.useState(args.click);
  
  const handleOpen = () => setOpen(!open);

  const handleExit = () => {
    setOpen(false)
  }
 
  return (
    <>
      <Button onClick={handleOpen}>Message Dialog</Button>
      <Dialog open={open} handler={handleOpen} className="border border-blue-800" style={{ width: '30%', height: '40%', margin: 'auto' }}>
        <div className="flex items-center justify-between ">
          <DialogHeader>Last courier Information</DialogHeader>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="mr-3 h-5 w-5 cursor-pointer"
            onClick={handleOpen}
            
          >
            <path
              fillRule="evenodd"
              d="M5.47 5.47a.75.75 0 011.06 0L12 10.94l5.47-5.47a.75.75 0 111.06 1.06L13.06 12l5.47 5.47a.75.75 0 11-1.06 1.06L12 13.06l-5.47 5.47a.75.75 0 01-1.06-1.06L10.94 12 5.47 6.53a.75.75 0 010-1.06z"
              clipRule="evenodd"
            />
          </svg>
        </div>
        <DialogBody divider className="ml-5">
          <div className="grid gap-6">
          <div className="text-gray-900"><strong>Item Type:</strong> Item 1</div>
          <div><strong>Item Status:</strong> Status</div>
          <div><strong>onTime:</strong> Time</div>
          </div>
        </DialogBody>
        <DialogFooter className="space-x-2">
          <Button variant="outlined" color="red" onClick={handleExit}>
            Exit
          </Button>
        </DialogFooter>
      </Dialog>
    </>
  );
}